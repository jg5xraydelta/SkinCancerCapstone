import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in (400, 600, 700):
    fm.fontManager.addfont(f"/home/claude/render/node_modules/@fontsource/source-sans-3/files/source-sans-3-latin-{w}-normal.woff2") if False else None
a = np.array(json.load(open(sys.argv[1]))) * 100; m, s = a.mean(0), a.std(0); ep = np.arange(1, a.shape[1] + 1)
BLUE, INK, MUTED, GRID = "#2a78d6", "#1A1A1A", "#3D4A55", "#DDE3E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.0, 2.3), dpi=600)
ax.fill_between(ep, m - s, m + s, color=BLUE, alpha=0.18, lw=0)
ax.plot(ep, m, color=BLUE, lw=1.4)
k = int(np.argmax(m)); ax.plot(ep[k], m[k], "o", ms=5, color=BLUE, mec="white", mew=1)
ax.annotate(f"chosen: epoch {ep[k]}\n{m[k]:.1f}% ± {s[k]:.1f}", (ep[k], m[k]), xytext=(ep[k] - 15, m[k] + 2.6),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlim(1, 30); ax.set_ylim(82, 94); ax.set_xticks([1, 10, 20, 30]); ax.set_yticks([82, 86, 90, 94])
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_xlabel("Epoch", color=MUTED); ax.set_ylabel("Val. accuracy (%)", color=MUTED)
fig.tight_layout(pad=0.3); fig.savefig(sys.argv[2], facecolor="white")
