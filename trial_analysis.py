import sys
import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "ee_individual_trials_raw.csv"
LS = [4, 8, 16]
df = pd.read_csv(CSV)


def paired_ci(diff, conf=0.95):
    n = len(diff)
    m = float(np.mean(diff))
    h = stats.sem(diff) * stats.t.ppf((1 + conf) / 2, n - 1)
    return m, m - h, m + h


def ols(X, y):
    """Plain OLS with 95% CIs. X must include a constant column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, t, r2


GDF_REPORTED = {"LOW (Manhattan Grid)": 1.9337, "MEDIUM (Paris Core)": 2.3571,
                "HIGH (Pittsburgh Rivers)": 2.9704}
BAND = (1000.0, 3000.0) 

for tier, g in df.groupby("tier_name", sort=False):
    print("\n" + "=" * 70)
    print(tier, f"(n = {len(g)})")
    print("=" * 70)

    # Route-length desc.
    print(f"Mean Euclidean OD distance : {g.euclidean_dist_m.mean():8.1f} m")
    print(f"Mean shortest-path cost    : {g.shortest_path_cost_s.mean():8.1f} s")
    eff = (g.euclidean_dist_m / g.shortest_path_cost_s) * 3.6   # km/h straight-line distance per travel time
    print(f"Effective straight-line speed d/t (mean of pairs): {eff.mean():5.1f} km/h")
    if tier in GDF_REPORTED:   # GDF = v_max * mean(t/d)  ->  v_max = GDF / mean(t/d)   (assumes same OD pairs as GDF sample)
        vmax = GDF_REPORTED[tier] / (g.shortest_path_cost_s / g.euclidean_dist_m).mean() * 3.6
        print(f"Implied v_max from reported GDF: {vmax:5.1f} km/h")

    # Per-node cost (microseconds)
    print("\nPer-expanded-node cost (mean runtime / mean nodes):")
    pn = {"Euclid": 1000 * g.euclidean_time_ms.mean() / g.euclidean_nodes.mean()}
    for L in LS:
        pn[f"ALT{L}"] = 1000 * g[f"alt_L{L}_time_ms"].mean() / g[f"alt_L{L}_nodes"].mean()
    for k, v in pn.items():
        print(f"  {k:7s}: {v:6.2f} us/node")
    slope, intercept, r, *_ = stats.linregress(LS, [pn[f"ALT{L}"] for L in LS])
    print(f"  ALT per-node cost ~ {intercept:.2f} + {slope:.2f} * L  (r^2 = {r**2:.3f})")
    print("  Break-even node ratio N_ALT/N_Euc = t_Euc/t_ALT :")
    for L in LS:
        be = pn["Euclid"] / pn[f"ALT{L}"]
        obs = g[f"alt_L{L}_nodes"].mean() / g.euclidean_nodes.mean()
        print(f"    L={L:2d}: break-even {be:.2f} | observed {obs:.2f} -> "
              f"{'ALT faster' if obs < be else 'Euclid faster'}")

    # Within-city regression: detour vs search effort
    cost_per_m = g.shortest_path_cost_s / g.euclidean_dist_m
    rel_detour = cost_per_m / cost_per_m.mean()
    print("\nWithin-city regression:  log(nodes) = b0 + b1*log(Euclid dist) + b2*log(rel. detour)")
    X = np.column_stack([np.ones(len(g)), np.log(g.euclidean_dist_m), np.log(rel_detour)])
    cols = {"Euclid": "euclidean_nodes", **{f"ALT{L}": f"alt_L{L}_nodes" for L in LS}}
    for name, col in cols.items():
        beta, se, t, r2 = ols(X, np.log(g[col].values.astype(float)))
        print(f"  {name:7s}: b1 = {beta[1]:5.2f} +/- {t*se[1]:.2f} | "
              f"b2 (detour) = {beta[2]:5.2f} +/- {t*se[2]:.2f} | R^2 = {r2:.2f}")
    print("  Node ratio ALT/Euclid vs detour (Spearman):")
    for L in LS:
        ratio = g[f"alt_L{L}_nodes"] / g.euclidean_nodes
        rho, p = stats.spearmanr(rel_detour, ratio)
        print(f"    L={L:2d}: rho = {rho:5.2f}, p = {p:.3g}")

    # Paired tests
    print("\nPaired runtime differences (ms), mean [95% CI], Wilcoxon p:")
    comps = [("Euclid", "euclidean_time_ms", f"ALT{L}", f"alt_L{L}_time_ms") for L in LS]
    comps += [("ALT4", "alt_L4_time_ms", "ALT8", "alt_L8_time_ms"),
              ("ALT8", "alt_L8_time_ms", "ALT16", "alt_L16_time_ms")]
    for a, ca, b, cb in comps:
        d = (g[ca] - g[cb]).values
        m, lo, hi = paired_ci(d)
        p = stats.wilcoxon(d).pvalue
        print(f"  {a:6s} - {b:5s}: {m:6.3f} [{lo:6.3f}, {hi:6.3f}]  p = {p:.3g}")
    print("\nSpeedup per L (ratio of mean runtimes, Euclid / ALT):")
    for L in LS:
        print(f"  L={L:2d}: {g.euclidean_time_ms.mean() / g[f'alt_L{L}_time_ms'].mean():.2f}x")


# Distance-matched comparison: only OD pairs whose Euclidean distance lies in a common band
print("\n" + "=" * 70)
print(f"DISTANCE-MATCHED COMPARISON (pairs with {BAND[0]:.0f} m <= Euclidean distance < {BAND[1]:.0f} m)")
print("=" * 70)
b = df[(df.euclidean_dist_m >= BAND[0]) & (df.euclidean_dist_m < BAND[1])]
print(f"{'Tier':26s}{'n':>5s}{'meanD(m)':>10s}{'Euc N':>8s}{'Euc ms':>8s}" +
      "".join(f"{'A'+str(L)+' N':>8s}{'ms':>6s}{'x':>6s}" for L in LS))
for tier, g in b.groupby("tier_name", sort=False):
    line = f"{tier:26s}{len(g):5d}{g.euclidean_dist_m.mean():10.0f}{g.euclidean_nodes.mean():8.1f}{g.euclidean_time_ms.mean():8.2f}"
    for L in LS:
        line += (f"{g[f'alt_L{L}_nodes'].mean():8.1f}{g[f'alt_L{L}_time_ms'].mean():6.2f}"
                 f"{g.euclidean_time_ms.mean() / g[f'alt_L{L}_time_ms'].mean():6.2f}")
    print(line)
print("Node reduction vs Euclid (%):")
for tier, g in b.groupby("tier_name", sort=False):
    print("  " + f"{tier:26s}" + "  ".join(f"L={L}: {100*(1-g[f'alt_L{L}_nodes'].mean()/g.euclidean_nodes.mean()):5.1f}" for L in LS))
