"""
selfa_delay_asymmetry_agency.py
================================
New model: Agency attribution arises from inter-hemispheric delay asymmetry

Core claim:
  Δτ = τ_R − τ_L = 0  →  A(t) ≡ 0  (proven analytically, confirmed numerically)
  Δτ > 0               →  A(t) > 0  (simulation)

Both hemispheres have access to motor commands.
The ONLY structural difference is the internal processing delay.

Outputs → ~/Desktop/results/agency_delay/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, ttest_1samp

# ── output ────────────────────────────────────────────────────────────────────
OUT = os.path.expanduser("~/Desktop/results/agency_delay")
os.makedirs(OUT, exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
N_RUNS     = 50
T          = 3000
T_BURN     = 300
T_SELF_END = T // 2        # first half = self-generated

TAU_WORLD  = 4             # true generative delay
TAU_L      = 4             # left hemisphere internal delay (fast, matched)
K          = 1.0           # true gain
RHO        = 0.8           # AR coefficient of environmental disturbance
SIGMA_E    = 0.5
SIGMA_XI   = 0.3
ALPHA      = 0.2           # attenuation in self-generated interval
ETA        = 0.01          # learning rate (identical for both hemispheres)

# Δτ sweep: τ_R = τ_L + Δτ
DELTA_TAU_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10]

# τ_world sweep (robustness, fixed Δτ=1)
TAU_WORLD_VALUES = list(range(0, 10))


# ═════════════════════════════════════════════════════════════════════════════
# Core simulation
# ═════════════════════════════════════════════════════════════════════════════

def run(seed, delta_tau, tau_world=TAU_WORLD):
    """
    Both hemispheres receive motor commands but with different internal delays.
    Left:  τ_L  (fast)
    Right: τ_R = τ_L + Δτ  (slow)

    Returns post-burn-in arrays: eps_L, eps_R, A, logBF, interval
    """
    rng   = np.random.default_rng(seed)
    tau_R = TAU_L + delta_tau
    pad   = max(TAU_L, tau_R, tau_world) + 1

    m  = rng.standard_normal(T + pad)   # motor command (white Gaussian)
    e  = np.zeros(T)
    s  = np.zeros(T)
    eL = np.zeros(T)
    eR = np.zeros(T)

    wL = 0.0   # left hemisphere gain
    wR = 0.0   # right hemisphere gain
    sigma2 = 1.0

    for t in range(T):
        # environmental disturbance
        e[t] = RHO * e[t-1] + rng.normal(0, SIGMA_E) if t > 0 else rng.normal(0, SIGMA_E)
        e_eff = ALPHA * e[t] if t < T_SELF_END else e[t]

        # sensory input
        s[t] = K * m[t - tau_world + pad] + e_eff + rng.normal(0, SIGMA_XI)

        # left hemisphere (delay = τ_L)
        mL    = m[t - TAU_L + pad]
        yL    = wL * mL
        eL[t] = s[t] - yL
        wL   += ETA * eL[t] * mL

        # right hemisphere (delay = τ_R = τ_L + Δτ)
        mR    = m[t - tau_R + pad]
        yR    = wR * mR
        eR[t] = s[t] - yR
        wR   += ETA * eR[t] * mR

        # running sigma^2 estimate
        sigma2 = 0.999 * sigma2 + 0.001 * s[t]**2
        sigma2 = max(sigma2, 1e-8)

    sl       = slice(T_BURN, T)
    A        = np.abs(eR[sl]) - np.abs(eL[sl])
    eps_bar  = 0.5 * (np.abs(eR[sl]) + np.abs(eL[sl]))
    logBF    = (eR[sl]**2 - eL[sl]**2) / (2 * sigma2)
    interval = np.array(["self" if t < T_SELF_END else "ext"
                         for t in range(T_BURN, T)])

    return dict(eps_L=eL[sl], eps_R=eR[sl], A=A, logBF=logBF,
                eps_bar=eps_bar, interval=interval,
                wL_final=wL, wR_final=wR)


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1: Δτ sweep — main result
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 1: Δτ sweep...")

results_dt = {}
for dt in DELTA_TAU_VALUES:
    runs = [run(s, dt) for s in range(N_RUNS)]
    # self-generated interval
    A_self = np.array([r["A"][r["interval"] == "self"].mean() for r in runs])
    A_ext  = np.array([r["A"][r["interval"] == "ext"].mean()  for r in runs])
    # log BF
    LBF    = np.array([r["logBF"][r["interval"] == "self"].mean() for r in runs])
    # t-test: A > 0?
    tstat, pval = ttest_1samp(A_self, 0)
    results_dt[dt] = dict(
        A_self_mean=A_self.mean(), A_self_sd=A_self.std(),
        A_ext_mean=A_ext.mean(),   A_ext_sd=A_ext.std(),
        LBF_mean=LBF.mean(),       LBF_sd=LBF.std(),
        t=tstat, p=pval,
        A_self_all=A_self,
    )
    print(f"  Δτ={dt:2d}  A_self={A_self.mean():.4f}±{A_self.std():.4f}"
          f"  p={pval:.3e}")

# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2: Δτ=0 → A ≡ 0  (strict numerical verification)
# ═════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 2: Δτ=0 degeneracy check...")

A0_vals = []
for s in range(N_RUNS):
    r = run(s, 0)
    A0_vals.append(np.abs(r["A"]).max())   # max absolute value across all t
print(f"  max|A(t)| over all t, all runs: {np.max(A0_vals):.2e}")
print(f"  (should be ≈ 0 up to floating-point precision)")

# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3: τ_world robustness  (Δτ=1, vary τ_world)
# ═════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 3: τ_world robustness (Δτ=1)...")

results_tw = {}
for tw in TAU_WORLD_VALUES:
    runs = [run(s, 1, tau_world=tw) for s in range(N_RUNS)]
    A_self = np.array([r["A"][r["interval"] == "self"].mean() for r in runs])
    results_tw[tw] = dict(mean=A_self.mean(), sd=A_self.std())
    print(f"  τ_world={tw}  A_self={A_self.mean():.4f}±{A_self.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Analysis 4: A(t) vs log BF  (Δτ=1)
# ═════════════════════════════════════════════════════════════════════════════
print("\nAnalysis 4: A(t) vs log BF correlation (Δτ=1)...")

A_cat   = np.concatenate([run(s, 1)["A"]     for s in range(N_RUNS)])
LBF_cat = np.concatenate([run(s, 1)["logBF"] for s in range(N_RUNS)])
r_corr, p_corr = pearsonr(A_cat, LBF_cat)
print(f"  r = {r_corr:.4f}, p = {p_corr:.2e}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure: 4-panel main figure
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

DT_VALS  = sorted(results_dt.keys())
A_means  = [results_dt[d]["A_self_mean"] for d in DT_VALS]
A_sds    = [results_dt[d]["A_self_sd"]   for d in DT_VALS]
LBF_m    = [results_dt[d]["LBF_mean"]    for d in DT_VALS]
P_VALS   = [results_dt[d]["p"]           for d in DT_VALS]

# ── Panel A: Δτ vs mean agency index ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.bar(DT_VALS, A_means, yerr=A_sds, capsize=4,
       color=["#d62728" if m <= 1e-6 else "#1f77b4" for m in A_means],
       edgecolor="k", linewidth=0.6, width=0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Δτ  (τ_R − τ_L)", fontsize=11)
ax.set_ylabel("Mean agency index ⟨A⟩  (self interval)", fontsize=10)
ax.set_title("(A)  Δτ = 0 → A ≡ 0;  Δτ > 0 → A > 0", fontsize=10, fontweight="bold")
# significance markers
for i, (dt, p) in enumerate(zip(DT_VALS, P_VALS)):
    if p < 0.001 and A_means[i] > 1e-6:
        ax.text(dt, A_means[i] + A_sds[i] + 0.005, "***",
                ha="center", fontsize=8)
ax.text(0, -0.015, "A ≡ 0", ha="center", color="#d62728", fontsize=8, style="italic")

# ── Panel B: Δτ=0 degeneracy time-series ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
r0 = run(0, 0)
r1 = run(0, 1)
t_ax = np.arange(len(r0["A"]))
ax.plot(t_ax, r0["A"], color="#d62728", lw=0.8, alpha=0.9, label="Δτ = 0")
ax.plot(t_ax, r1["A"], color="#1f77b4", lw=0.8, alpha=0.7, label="Δτ = 1")
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.axvline(T_SELF_END - T_BURN, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(T_SELF_END - T_BURN + 10, ax.get_ylim()[0] * 0.85,
        "← external", fontsize=8, color="gray")
ax.set_xlabel("Time step", fontsize=11)
ax.set_ylabel("A(t)", fontsize=11)
ax.set_title("(B)  Time series: Δτ=0 vs Δτ=1", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

# ── Panel C: τ_world robustness ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
tw_list = sorted(results_tw.keys())
tw_means = [results_tw[t]["mean"] for t in tw_list]
tw_sds   = [results_tw[t]["sd"]   for t in tw_list]
ax.errorbar(tw_list, tw_means, yerr=tw_sds,
            fmt="o-", color="#2ca02c", capsize=4, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.axvline(TAU_WORLD, color="gray", lw=1.0, ls=":", alpha=0.7,
           label=f"τ_world = τ_L = {TAU_WORLD}")
ax.set_xlabel("τ_world", fontsize=11)
ax.set_ylabel("Mean agency index ⟨A⟩  (self interval)", fontsize=10)
ax.set_title("(C)  Robustness: vary τ_world  (Δτ = 1)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)

# ── Panel D: A(t) vs log BF scatter ──────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
rng_plot = np.random.default_rng(42)
idx = rng_plot.choice(len(A_cat), size=min(6000, len(A_cat)), replace=False)
ax.scatter(A_cat[idx], LBF_cat[idx],
           s=2, alpha=0.25, color="#9467bd", rasterized=True)
xl = np.percentile(np.abs(A_cat), 98)
ax.set_xlim(-xl, xl)
ax.set_ylim(np.percentile(LBF_cat, 1), np.percentile(LBF_cat, 99))
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.axvline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel("Agency index A(t)", fontsize=11)
ax.set_ylabel("Instantaneous log BF(t)", fontsize=11)
ax.set_title(f"(D)  A(t) ∝ log BF(t),  r = {r_corr:.3f}",
             fontsize=10, fontweight="bold")

fig.suptitle(
    "Inter-hemispheric delay asymmetry (Δτ) as the necessary condition for agency attribution",
    fontsize=12, fontweight="bold", y=1.01
)

for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"fig_agency_delay_main.{ext}"),
                dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT}/fig_agency_delay_main.pdf/.png")


# ═════════════════════════════════════════════════════════════════════════════
# CSV outputs
# ═════════════════════════════════════════════════════════════════════════════

rows = []
for dt in DT_VALS:
    d = results_dt[dt]
    rows.append(dict(
        delta_tau=dt,
        A_self_mean=d["A_self_mean"], A_self_sd=d["A_self_sd"],
        A_ext_mean=d["A_ext_mean"],   A_ext_sd=d["A_ext_sd"],
        logBF_mean=d["LBF_mean"],     logBF_sd=d["LBF_sd"],
        t_stat=d["t"], p_val=d["p"],
    ))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_delta_tau_sweep.csv"), index=False)

rows2 = [dict(tau_world=tw, A_self_mean=results_tw[tw]["mean"],
              A_self_sd=results_tw[tw]["sd"]) for tw in tw_list]
pd.DataFrame(rows2).to_csv(os.path.join(OUT, "table_tau_world_robustness.csv"), index=False)

print("CSVs saved.")
print("\n=== KEY RESULTS ===")
print(f"Δτ=0  max|A| = {np.max(A0_vals):.2e}  (numerical zero)")
print(f"Δτ=1  ⟨A⟩   = {results_dt[1]['A_self_mean']:.4f} ± {results_dt[1]['A_self_sd']:.4f}"
      f"  p = {results_dt[1]['p']:.2e}")
print(f"A(t) vs log BF:  r = {r_corr:.4f}")
