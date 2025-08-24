
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import re

import ast



def plot_compare_linePlot(df, metrics):
    feedback_types = df["feedback_type"].unique()
    n = len(feedback_types)

    for metric in metrics:
        # Create figure with n rows: one per feedback type
        fig, axs = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=False)
        fig.suptitle(f"{metric.replace('_', ' ').title()} — by Feedback Type", fontsize=16)

        for i, feedback_type in enumerate(feedback_types):
            # LINE PLOT
            ax_line = axs[i, 0]
            for (source, expert_type), group in df[df["feedback_type"] == feedback_type].groupby(["source", "expert_type"]):
                label = f"{source}: {expert_type}"
                ax_line.plot(group["delta"], group[metric], marker="o", label=label)
            ax_line.set_title(f"{feedback_type} — Line Plot")
            ax_line.set_ylabel(metric.replace('_', ' ').title())
            ax_line.set_xlabel("Delta")
            ax_line.grid(True)
            if i == 0:
                ax_line.legend(title="Expert Type", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

            # HEATMAP
            ax_heat = axs[i, 1]
            pivot = df[df["feedback_type"] == feedback_type].pivot_table(
                index="expert_type", columns="delta", values=metric, aggfunc="mean"
            )
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax_heat)
            ax_heat.set_title(f"{feedback_type} — Heatmap")
            ax_heat.set_xlabel("Delta")
            ax_heat.set_ylabel("Expert Type")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



def plot_timeseries(df, metrics):
    """
    df: DataFrame with columns [delta, expert_type, rewards_per_time, queries_per_time]
    deltas: list of delta values to plot
    metrics: which timeseries to plot
    """
    for delta in set(df["delta"]):

        for feedback_type in set(df["feedback_type"]):

            fig, axs = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 4))
            if len(metrics) == 1:
                axs = [axs]
            fig.suptitle(f"Δ = {delta}, Feedback_type = {feedback_type}", fontsize=14)

            for j, metric in enumerate(metrics):
                ax = axs[j]

                # for expert_type, group in df.groupby("expert_type"):
                #     series = np.vstack(group[metric].values)  # stack runs
                #     mean = series.mean(axis=0)
                #     std = series.std(axis=0)
                #     ax.plot(mean, label=expert_type)
                #     ax.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.2)

                for expert_type, group in df.groupby("expert_type"):
                    subset = group.loc[(group["delta"] == delta) & (group["feedback_type"] == feedback_type)]
                    ax.plot(subset[metric].iloc[0], label=expert_type)

                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel("time steps")
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()



def ensure_array(x):
    if isinstance(x, str):
        # remove "np.float64(" and the closing ")"
        cleaned = x.replace("np.float64(", "").replace(")", "").replace(" ","").replace("[","").replace("]","")
        # now extract numbers (handles floats and ints)
        x_list = cleaned.split(",")
        return np.array([float(n) for n in x_list], dtype=float)
    elif isinstance(x, list):
        return np.array(x, dtype=float)
    elif isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        raise ValueError(f"Unexpected type {type(x)}")




########################################################################################

# INIT:


# files = {
#     "OG": "MixUCB_compare_21Aug2025.xlsx",
#     "EXP4": "MixUCB_compare_EXP4_21Aug2025.xlsx",
#     "MetaChoice": "MixUCB_compare_MetaChoice_21Aug2025.xlsx"
# }
files = {
    "OG": "MixUCB_compare.xlsx",
    "EXP4": "MixUCB_compare_EXP4.xlsx",
    "MetaChoice": "MixUCB_compare_MetaChoice.xlsx"
}

metrics = ["cum_auto_rewards_per_time", "cum_queries_per_time"]


########################################################################################



# CODE:


# df = pd.read_excel("MixUCB_compare_Jul11.xlsx")

# Load and tag each DataFrame
dfs = []
for label, filepath in files.items():
    df = pd.read_excel(filepath)
    df["source"] = label  # Add source column
    dfs.append(df)

# Concatenate all into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)

#conversion step (because in excel list is saved as str):
for col in metrics:
    if col in df_all.columns:
        df_all[col] = df_all[col].apply(ensure_array)



# plot_compare_heatmap(df_all)

# plot_compare_linePlot(df_all, metrics=["total_reward", "query_num", "total_auto_reward"])

plot_timeseries(df_all, metrics)