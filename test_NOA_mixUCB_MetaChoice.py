# from generate_multilabel_data import generate_synthetic_data


# data = generate_synthetic_data(T=10, noise_std=0.2, seed=42)
# print(data)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_multilabel_data import generate_synthetic_data
from run_allucb_NOA_mixUCB import run_mixucb, run_linear_oracle, run_expert   # uses the repo’s run functions
from utils.regression_ucb import OnlineLogisticRegressionOracle    # regression + UCB online oracle


def plot_query_reward(
    query_nums,
    rewards,
    feedback_type,
    expert_type,
    delta,
    *,
    is_cumulative=True,     # set False if query_nums are per-step counts already
    smooth_window=20,       # rolling mean window for rewards
    show_query_markers=True
):
    """
    query_nums : array-like
        If is_cumulative=True, this is cumulative queries over time.
        If is_cumulative=False, this is per-step query counts.
    rewards : array-like
        Per-step auto reward (not pre-averaged).
    """

    q = np.asarray(query_nums, dtype=float).ravel()
    r = np.asarray(rewards, dtype=float).ravel()
    T = min(len(q), len(r))
    q, r = q[:T], r[:T]

    # derive cumulative & per-step queries consistently
    if is_cumulative:
        cum_q = q
        per_step_q = np.diff(np.r_[0.0, q])
    else:
        per_step_q = q
        cum_q = np.cumsum(q)

    # clamp any tiny negative diffs (numerical jitter) to zero
    per_step_q = np.maximum(per_step_q, 0.0)

    # rolling mean for rewards
    w = int(max(1, min(smooth_window, T)))
    kernel = np.ones(w) / w
    roll_mean = np.convolve(r, kernel, mode="same")

    # cumulative average reward
    cum_avg = np.cumsum(r) / np.arange(1, T + 1)

    x = np.arange(T)

    plt.figure(figsize=(12, 5))

    # -------- Left: queries --------
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, cum_q, label="Cumulative Queries")
    ax1.set_title("Cumulative & Per-Step Queries")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Cumulative")

    # ax1b = ax1.twinx()
    # ax1b.plot(x, per_step_q, linestyle="--", beta_sq=0.8, label="Per-Step Queries")
    # ax1b.set_ylabel("Per-Step")

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left")

    # -------- Right: rewards --------
    ax2 = plt.subplot(1, 2, 2)
    # ax2.plot(x, r, beta_sq=0.5, label="Reward (per step)")
    ax2.plot(x, roll_mean, linewidth=2, label=f"Rolling Mean (w={w})")
    ax2.plot(x, cum_avg, linewidth=2, linestyle=":", label="Cumulative Average Auto Reward")
    ax2.set_title("Auto Reward")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Reward")
    ax2.legend(loc="best")

    # optional: show vertical markers where a query happened
    if show_query_markers:
        query_idxs = np.where(per_step_q > 0)[0]
        for idx in query_idxs:
            ax2.axvline(idx, beta_sq=0.15)

    plt.suptitle(f"Feedback: {feedback_type}, Expert: '{expert_type}', Δ={delta}", fontsize=14)
    plt.tight_layout()
    plt.show()


########################################################################################################
########################################################################################################


# --- config (edit freely) ---
T = 300                        # timesteps to simulate (per run)
seed = 42                      # numpy RNG seed
dataset = "synthetic"          # "synthetic" | "spanet" | "heart_disease" | "MedNIST"
noise_std = 0.2                # only used for synthetic
deltas = [0.5, 0.75, 1.0]          # query thresholds Δ
beta_sq = 1.25                   # squared-loss radius (rad_sq)
beta_lr = 2.5                     # log-loss radius
lambda_ = 1e-3                 # L2 reg
lr = 0.1                       # learning rate

np.random.seed(seed)

# --- get data dict exactly like the repo expects ---
if dataset == "synthetic":
    data = generate_synthetic_data(T=1000, noise_std=noise_std, seed=seed)
elif dataset in ("heart_disease", "MedNIST", "yeast", "iris"):
    # You can also generate these through generate_multilabel_data.py,
    # but here we keep it simple and just re-use 'synthetic' for a quick demo.
    raise ValueError("For non-synthetic datasets, generate a pickle via generate_multilabel_data.py and load it.")
else:
    raise ValueError("Unknown dataset")

# Infer shapes from the data
n_actions = len(data["rounds"][0]["actual_rewards"])
n_features = data["rounds"][0]["context"].shape[1]

def avg_autonomous_reward(rewards: np.ndarray, queries: np.ndarray) -> float:
    """Average reward on steps where we did NOT query (Z_t = 0)."""
    mask_no_query = (queries == 0).astype(int)
    if mask_no_query.sum() == 0:
        return 0.0
    return (rewards * mask_no_query).sum() / mask_no_query.sum()



###############################################################################################

#NOA:


def phi(x, a, K):
    # Feature map
    ea = np.zeros(K)
    ea[a] = 1
    phi_val = np.kron(x, ea)
    return phi_val      # / np.linalg.norm(phi_val)  # normalize


actions_set = list(range(n_actions))

# expert_types = ["compare", "demonstrate", "improve", "reward_punish", "off"]
expert_types = ["improve", "reward_punish", "off"]

reward_list_dict = {
    # "compare": lambda rewards, context: (
    #     subset := oracle.get_ucb_lcb_subset(context, 2),
    #     [rewards[a] for a in subset if a in rewards]
    # ),
    # "demonstrate": lambda rewards, context: (
    #     subset := actions_set,
    #     [rewards[a] for a in subset if a in rewards]
    # ),
    "improve": lambda rewards, context: (
        subset := oracle.get_ucb_lcb_subset(context, 1) + [max(rewards, key=rewards.get)],
        [rewards[a] for a in subset if a in rewards]
    ),
    "reward_punish": lambda rewards, context: (
        subset := oracle.get_ucb_lcb_subset(context, 1) + [max(rewards, key=rewards.get)],
        [-1,1]  #-1 for robot, +1 for expert
    ),
    "off": lambda rewards, context: (
        subset := oracle.get_ucb_lcb_subset(context, 1) + [0],  #define external "off" action as: 0
        [rewards[a] for a in subset if a in rewards] 
    )
}


###############################################################################################




rows = []

# Baselines: perfect/noisy expert + linear oracles
# (shows how to call other helpers without touching the file system)
# perfect expert
# r_exp, a_exp = run_expert(data, T, type="perfect_exp")
# rows.append(dict(mode="perfect_exp", delta=None,
#                  queries=0, avg_auto_reward=0.0,
#                  avg_total_reward=r_exp.mean()))
# # noisy expert
# r_exp, a_exp = run_expert(data, T, type="noisy_exp")
# rows.append(dict(mode="noisy_exp", delta=None,
#                  queries=0, avg_auto_reward=0.0,
#                  avg_total_reward=r_exp.mean()))
# # squared/logistic “oracle” (uses data['true_theta*'])
# r_or, a_or = run_linear_oracle(data, T, data["true_theta"])
# rows.append(dict(mode="sq_oracle", delta=None,
#                  queries=0, avg_auto_reward=r_or.mean(),  # same as total here
#                  avg_total_reward=r_or.mean()))
# r_or, a_or = run_linear_oracle(data, T, data["true_theta_classification"])
# rows.append(dict(mode="lr_oracle", delta=None,
#                  queries=0, avg_auto_reward=r_or.mean(),
#                  avg_total_reward=r_or.mean()))




# MixUCB family (lin, mixI, mixII, mixIII)
for expert_type in expert_types:
    for mode in ["lin", "mixI", "mixII", "mixIII"]:
        # For 'lin' there is no Δ list in the repo – it internally treats Δ as [0] (no query path). 
        _deltas = [0.0] if mode == "lin" else deltas
        for delta in _deltas:
            oracle = OnlineLogisticRegressionOracle(
                n_features, n_actions, lr, lambda_, beta_lr, rad_sq=beta_sq
            )
            r_t, q_t, a_t = run_mixucb(
                data=data,
                T=T,
                n_actions=n_actions,
                n_features=n_features,
                reward_list_dict=reward_list_dict,
                expert_type=expert_type, 
                delta=delta,
                online_reg_oracle=oracle,
                mode=mode,
            )  # returns reward_per_time, query_per_time, action_per_time 

            rows.append(dict(
                mode=mode,
                delta=float(delta),
                queries=int(q_t.sum()),
                avg_auto_reward=avg_autonomous_reward(r_t, q_t),
                avg_total_reward=r_t.mean(),
            ))


            # PLOT:
            query_nums = [int(q_t[:i].sum()) for i in range(len(q_t))]
            total_auto_rewards = [avg_autonomous_reward(r_t[:i], q_t[:i]) for i in range(len(q_t))]
            feedback_type = mode
            expert_type = expert_type
            plot_query_reward(query_nums,                  # cumulative or per-step; set is_cumulative accordingly
                            total_auto_rewards,          # per-step rewards array
                            feedback_type,
                            expert_type,
                            delta,
                            is_cumulative=True,          # flip to False if passing per-step counts
                            smooth_window=25,            # tweak as you like
                            show_query_markers=False
                        )

df = pd.DataFrame(rows)
print(df)

# Save a compact summary like your Excel sheet idea
out_path = Path("MixUCB_compare.xlsx")
df.to_excel(out_path, index=False)
print(f"Saved results -> {out_path.resolve()}")

# (Optional) Tiny sanity plot
try:
    import matplotlib.pyplot as plt
    for mode in ["lin", "mixI", "mixII", "mixIII"]:
        sub = df[df["mode"] == mode]
        xs = sub["delta"].fillna(0.0)
        plt.plot(xs, sub["avg_total_reward"], marker="o", label=mode)
    plt.xlabel("Δ"); plt.ylabel("Avg total reward"); plt.title("Quick comparison")
    plt.legend(); plt.tight_layout(); plt.show()
except Exception as e:
    print("Plot skipped:", e)

