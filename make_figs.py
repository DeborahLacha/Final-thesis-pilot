import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "mathtext.fontset": "stix"})

res = json.load(open(sys.argv[1]))
out = sys.argv[2]
rho = np.array([r["rho"] for r in res])

fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.2), gridspec_kw=dict(wspace=0.5))

# (a) analytic reversal region for the three-feature example
a = ax[0]
r = np.linspace(0, 1, 200)
a.fill_between(r, 1 - r / 2, 1, color=BLUE, alpha=0.30, lw=0)
a.plot(r, 1 - r / 2, color=BLUE, lw=1.6)
a.plot(r, np.ones_like(r), color=BLUE, lw=1.6)
a.text(0.62, 0.86, "reversal", color=INK, fontsize=8, ha="center")
a.text(0.35, 0.45, "same ordering\n(no reversal)", color=MUTED, fontsize=7.5, ha="center")
a.text(0.35, 1.13, "same ordering", color=MUTED, fontsize=7.5, ha="center")
a.set_xlim(0, 1); a.set_ylim(0.3, 1.3)
a.set_xlabel(r"correlation $\rho$ of $X_1, X_2$"); a.set_ylabel(r"effect size $c$ of $X_3$")
a.set_title("(a) Example 3.1 (analytic)", fontsize=8.5, color=INK)

# (b) reversal rates vs correlation
b = ax[1]
b.plot(rho, 100 * np.array([x["reversal"] for x in res]), color=BLUE, lw=1.8, marker="o", ms=3.5,
       label="$p<0.05$")
b.plot(rho, 100 * np.array([x["practical_reversal"] for x in res]), color=ORANGE, lw=1.8, marker="s",
       ms=3.5, label="$p<0.05$, lead $>5\\%$")
b.set_xlabel(r"equicorrelation $\rho$"); b.set_ylabel("models with a reversal (%)")
b.set_ylim(-3, 103); b.set_xlim(-0.02, 0.97)
b.set_title("(b) Pilot: reversal rate", fontsize=8.5, color=INK)
b.text(0.15, 86, "$p<0.05$", color=BLUE, fontsize=7.5)
b.text(0.47, 42, "$p<0.05$ and\nlead $>5\\%$", color=ORANGE, fontsize=7.5)

# (c) size of the home-operator margin
c = ax[2]
c.plot(rho, 100 * np.array([x["margin_marg"] for x in res]), color=BLUE, lw=1.8, marker="o", ms=3.5,
       label="Int-SHAP lead, marginal")
c.plot(rho, 100 * np.array([x["margin_cond"] for x in res]), color=ORANGE, lw=1.8, marker="s", ms=3.5,
       label="Cond-SHAP lead, conditional")
c.set_xlabel(r"equicorrelation $\rho$"); c.set_ylabel("median lead (% of range)")
c.set_ylim(0, 92); c.set_xlim(-0.02, 0.97)
c.set_title("(c) Pilot: size of the lead", fontsize=8.5, color=INK)
c.legend(frameon=False, fontsize=6.8, loc="upper left")

for x in ax:
    x.grid(True, color=GRID, lw=0.5); x.set_axisbelow(True)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
