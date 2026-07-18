# API_MAPPING — R (DeclareDesign ecosystem) → Python targets

One entry per course-load-bearing function. "Course-inline" = the pattern the
notebooks actually ship (transparent, narrated, no package); "parity target" =
what the eventual package would expose. See `PARITY_MATRIX.csv` for status.

## The design pipeline

### `declare_model(N, ...)` → dataframe factory

```r
declare_model(N = 100, U = rnorm(N),
              potential_outcomes(Y ~ 0.2 * Z + U))
```

```python
def make_world(n=100, effect=0.2, rng=rng):
    U = rng.normal(size=n)
    return pd.DataFrame({"U": U, "Y0": U, "Y1": effect + U})
```

Course-inline: potential outcomes are explicit columns (`Y0`, `Y1`) — the
pedagogy wants the fundamental problem visible. Parity target: a composable
`Model` step object.

### `declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0))` → truth on the population

```python
true_ate = (world["Y1"] - world["Y0"]).mean()
```

### `declare_sampling(S = complete_rs(N, n))` / `simple_rs` → seeded pandas

```python
sample = world.sample(n=100, random_state=464)          # complete_rs
mask   = rng.random(len(world)) < 0.10                  # simple_rs
```

### `declare_assignment(Z = complete_ra(N, m))` → seeded permutation

```python
def complete_ra(n, m, rng):
    z = np.zeros(n, dtype=int); z[:m] = 1
    return rng.permutation(z)
```

### `reveal_outcomes(Y ~ Z)` → one-line switch

```python
df["Y"] = np.where(df["Z"] == 1, df["Y1"], df["Y0"])
```

### `declare_measurement(M = latent + rnorm(N, 0, s))` → noise/bias injection

```python
df["M_noisy"]  = df["latent"] + rng.normal(0, s, len(df))   # random error
df["M_biased"] = df["latent"] + b                            # systematic error
```

## The answer strategies

### `difference_in_means(Y ~ Z, data)` → hand-rolled + interval

```python
def difference_in_means(df, y="Y", z="Z"):
    g1, g0 = df.loc[df[z] == 1, y], df.loc[df[z] == 0, y]
    est = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1)/len(g1) + g0.var(ddof=1)/len(g0))
    return est, se
```

(estimatr's design-aware SE refinements — blocked/clustered — are parity
targets, not course scope.)

### `lm_robust(Y ~ Z + X, data)` → statsmodels with HC2

```python
import statsmodels.formula.api as smf
fit = smf.ols("Y ~ Z + X", data=df).fit(cov_type="HC2")
```

HC2 is estimatr's default — the course keeps that default so numbers line up
with the book's outputs.

## The diagnosis engine

### `diagnose_design(design, sims=1000)` → seeded Monte-Carlo loop

```python
def diagnose(design_fn, truth_fn, reps=1000, seed=464, alpha=0.05):
    rng = np.random.default_rng(seed)
    rows = [design_fn(rng) for _ in range(reps)]      # each: est, se, truth
    d = pd.DataFrame(rows, columns=["est", "se", "truth"])
    return pd.Series({
        "bias":     (d.est - d.truth).mean(),
        "power":    (np.abs(d.est / d.se) > 1.96).mean(),
        "coverage": ((d.est - 1.96*d.se <= d.truth) &
                     (d.truth <= d.est + 1.96*d.se)).mean(),
    })
```

### `redesign(design, n = c(100, 200, 400))` → grid rerun

```python
comparison = pd.DataFrame({n: diagnose(make_design(n), ...) for n in (100, 200, 400)}).T
```

## Utilities

### `make_dag_df()` → networkx

```python
import networkx as nx
g = nx.DiGraph([("X", "Y"), ("U", "X"), ("U", "Y")])
nx.draw_networkx(g, pos=nx.spring_layout(g, seed=464))
```

## Datasets

`rdss::lapop_brazil` etc. → shipped CSVs in `notebooks/data/` (MIT +
attribution) — no dataverse client needed.
