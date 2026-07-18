"""Priority book figures, recomputed and redrawn with declarepy (Tranche 4).

Re-implements the *message* of the course-priority figure families from
*Research Design in the Social Sciences* (chs. 2, 9, 10, 11, 18) in
matplotlib, on top of freshly diagnosed declarepy translations. Each figure
carries a structural check: the plotted numbers ARE the diagnosand tables
the validation suite checks against the book's reference outputs.

Run:  .venv/bin/python examples/book_figures.py     (writes examples/figures/*.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import declarepy as dp
from declarepy import viz
from declarepy.library import (
    declaration_2_1,
    declaration_9_1,
    declaration_10_1,
    declaration_11_1,
    declaration_18_9_encouragement,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
SIMS = 1000
SEED = 464


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close("all")
    print(f"✓ {name}")


# Fig. 2-family: does the program "succeed" as the confounder strength grows?
success = dp.Diagnosands(
    success=lambda d: float(
        ((d["estimate"] > 0.3) & (d["p_value"] < 0.05) & (d["estimand"] > 0.2)).mean()
    ),
)
grid_b = dp.redesign(declaration_2_1, b=[0.0, 0.75, 1.5, 2.25, 3.0])
diag_b = dp.diagnose_all(grid_b, sims=SIMS, seed=SEED, diagnosands=success).diagnosands
assert diag_b["success"].between(0, 1).all()
viz.plot_sweep(diag_b, x="b", y="success")
plt.title("Declaration 2.1 — probability the program is judged a success, by b")
save("fig02_success_by_b.png")

# Fig. 9-family: the n=3 answer strategy's wild sampling distribution.
diag9 = dp.diagnose(declaration_9_1(), sims=SIMS, seed=SEED)
viz.plot_sampling_distribution(diag9.simulations)
plt.title("Declaration 9.1 — mean age estimated from a sample of 3")
save("fig09_sampling_distribution.png")

# Fig. 10-family: the coverage picture for the canonical two-arm trial.
diag10 = dp.diagnose(declaration_10_1(), sims=SIMS, seed=SEED)
viz.plot_ci_caterpillar(diag10.simulations, n=100)
save("fig10_ci_caterpillar.png")

# Fig. 11-family: power over N for the redesign workhorse.
grid11 = dp.redesign(declaration_11_1, N=list(range(100, 1001, 100)))
diag11 = dp.diagnose_all(grid11, sims=SIMS, seed=SEED).diagnosands
powers = diag11.sort_values("N")["power"].to_numpy()
assert (np.diff(powers) > -0.05).all(), "power should rise with N (within MC noise)"
viz.plot_power_curve(diag11, x="N")
plt.title("Declaration 11.1 — power to detect the unfair coin, by N")
save("fig11_power_by_n.png")

# Fig. 11/18 grid idiom: two-parameter sweep as a heatmap.
grid_np = dp.redesign(
    declaration_10_1, N=[50, 100, 200, 400], effect=[0.1, 0.2, 0.4]
)
diag_np = dp.diagnose_all(grid_np, sims=SIMS, seed=SEED).diagnosands
viz.plot_grid_heatmap(diag_np, x="N", y="effect", value="power")
plt.title("Two-arm trial — power over the N × effect grid")
save("fig11_power_heatmap.png")

# Fig. 18-family: encouragement design, 2SLS power over compliance.
grid18 = dp.redesign(
    declaration_18_9_encouragement, compliance_rate=[0.1, 0.3, 0.5, 0.7, 0.9]
)
diag18 = dp.diagnose_all(grid18, sims=SIMS, seed=SEED).diagnosands
cace = diag18[diag18["inquiry"] == "CACE"] if "inquiry" in diag18 else diag18
viz.plot_power_curve(cace, x="compliance_rate")
plt.title("Declaration 18.9 (encouragement) — CACE power by compliance rate")
save("fig18_power_by_compliance.png")

print(f"\nAll figures written to {OUT}/")
