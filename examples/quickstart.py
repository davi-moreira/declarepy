"""declarepy quickstart: declare → diagnose → redesign in ~40 lines.

Run:  .venv/bin/python examples/quickstart.py
"""

import numpy as np

import declarepy as dp


# 1) DECLARE a two-arm trial as a factory of its parameters (M, I, D, A).
def two_arm_trial(N: int = 100, effect: float = 0.2) -> dp.Design:
    return dp.Design(
        dp.Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}),
        dp.potential_outcomes(
            lambda df, z, rng: effect * float(z) + df["U"].to_numpy()
        ),
        dp.Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        dp.Assignment.complete(prob=0.5),
        dp.reveal_outcomes(),  # Y = np.where(Z == 1, Y1, Y0) — visible switch
        dp.Estimator.lm_robust("Y ~ Z", inquiry="ATE"),
    )


design = two_arm_trial()

# One simulated study: the data, the truth, and the estimate.
run = dp.run_design(design, rng=464)
print("estimand:", run.estimands)
print(run.estimates[["estimator", "term", "estimate", "std_error", "p_value"]])

# 2) DIAGNOSE: 1,000 imagined studies, summarized.
diagnosis = dp.diagnose(design, sims=1000, seed=464)
print("\nDiagnosis (bias / power / coverage ...):")
print(diagnosis.diagnosands.round(3).to_string(index=False))

# 3) REDESIGN: how big must the study be?
grid = dp.redesign(two_arm_trial, N=[100, 400, 1600], effect=0.2)
sweep = dp.diagnose_all(grid, sims=1000, seed=464)
print("\nPower by sample size:")
print(sweep.diagnosands[["design", "N", "power"]].round(3).to_string(index=False))

rng = np.random.default_rng(464)
print("\nProcedures compose too:", dp.complete_ra(10, 5, rng=rng))
