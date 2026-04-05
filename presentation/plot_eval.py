"""Generate evaluation bar chart for the presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
labels   = ["No-Agent\n(Plain LLM)", "Agent\n(With Tools)"]
values   = [77, 83]
colors   = ["#93c5fd", "#1d4ed8"]   # light blue, dark blue

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
fig.patch.set_facecolor("white")

bars = ax.bar(labels, values, color=colors, width=0.45,
              edgecolor="white", linewidth=1.5,
              zorder=3)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val}%",
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color="#1e293b")

# Improvement annotation
ax.annotate("",
            xy=(1, 83), xytext=(0, 77),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2))
ax.text(0.5, 80.5, "+6%", ha="center", va="bottom",
        fontsize=13, fontweight="bold", color="#1d4ed8")

# Styling
ax.set_ylim(70, 88)
ax.set_ylabel("Accuracy (%)", fontsize=13, color="#334155")
ax.set_title("TriviaQA Evaluation — 100 Questions",
             fontsize=14, fontweight="bold", color="#1e293b", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cbd5e1")
ax.spines["bottom"].set_color("#cbd5e1")
ax.tick_params(colors="#334155", labelsize=12)
ax.set_facecolor("#f8fafc")

plt.tight_layout()
plt.savefig("eval_chart.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("Saved eval_chart.png")
