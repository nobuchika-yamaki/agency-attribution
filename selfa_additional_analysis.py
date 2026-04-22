"""
selfa_additional_analysis.py
=============================
Additional analyses for the delay-asymmetry agency model.

1. 2D sweep: Δτ × τ_world  → agency index heatmap
2. Agency direction map: sign(A) as function of (Δτ, τ_world)
3. Abruptness quantification: transition sharpness at Δτ=0
4. Noise robustness: σ_ξ and σ_e sweeps
5. Learning rate robustness: η sweep

Outputs → ~/Desktop/results/agency_delay/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter

OUT = os.path.expanduser("~/Desktop/results/agency_delay")
os.makedirs(OUT, exist_ok=True)

# ── base parameters (same as main simulation) ─────────────────────────────────
N_RUNS     = 30          # slightly fewer for 2D sweep speed
T          = 2000
T_BURN     = 200
T_SELF_END = T // 2
TAU_L      = 4
K          = 1.0
RHO        = 0.8
SIGMA_E    = 0.5
SIGMA_XI   = 0.3
ALPHA      = 0.2
ETA        = 0.01

# sweep ranges
DELTA_TAU_2D   = [0, 1, 2, 3, 4, 6, 8, 10]
TAU_WORLD_2D   = list(range(0, 12))

DELTA_TAU_FINE = [0, 1, 2, 3, 4, 5, 6, 8, 10]   # for abruptness
SIGMA_XI_VALS  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
SIGMA_E_VALS   = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
ETA_VALS       = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]


# ═════════════════════════════════════════════════════════════════════════════
# Core simulation (copied for self-containment)
# ═════════════════════════════════════════════════════════════════════════════

def run(seed, delta_tau, tau_world=4,
        sigma_xi=SIGMA_XI, sigma_e=SIGMA_E, eta=ETA):
    rng   = np.random.default_rng(seed)
    tau_R = TAU_L + delta_tau
    pad   = max(TAU_L, tau_R, tau_world) + 1

    m  = rng.standard_normal(T + pad)
    e  = np.zeros(T)
    eL = np.zeros(T)
    eR = np.zeros(T)
    wL = wR = 0.0
    sigma2 = 1.0

    for t in range(T):
        e[t]   = RHO * e[t-1] + rng.normal(0, sigma_e) if t > 0 else rng.normal(0, sigma_e)
        e_eff  = ALPHA * e[t] if t < T_SELF_END else e[t]
        st     = K * m[t - tau_world + pad] + e_eff + rng.normal(0, sigma_xi)

        mL     = m[t - TAU_L  + pad]
        mR     = m[t - tau_R  + pad]
        eL[t]  = st - wL * mL;  wL += eta * eL[t] * mL
        eR[t]  = st - wR * mR;  wR += eta * eR[t] * mR
        sigma2 = max(0.999 * sigma2 + 0.001 * st**2, 1e-8)

    sl = slice(T_BURN, T)
    A  = np.abs(eR[sl]) - np.abs(eL[sl])
    iv = np.array(["self" if t < T_SELF_END else "ext"
                   for t in range(T_BURN, T)])
    return dict(A=A, interval=iv)


def mean_A_self(seed_list, delta_tau, **kw):
    vals = []
    for s in seed_list:
        r = run(s, delta_tau, **kw)
        vals.append(r["A"][r["interval"] == "self"].mean())
    return np.array(vals)


seeds = list(range(N_RUNS))


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1: 2D heatmap  Δτ × τ_world
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 1: 2D sweep Δτ × τ_world ...")

hmap_mean = np.zeros((len(DELTA_TAU_2D), len(TAU_WORLD_2D)))
hmap_sign = np.zeros_like(hmap_mean)

for i, dt in enumerate(DELTA_TAU_2D):
    for j, tw in enumerate(TAU_WORLD_2D):
        vals = mean_A_self(seeds, dt, tau_world=tw)
        hmap_mean[i, j] = vals.mean()
        hmap_sign[i, j] = np.sign(vals.mean())
        print(f"  Δτ={dt} τ_w={tw}  A={vals.mean():.4f}", end="\r")

print("\n  Done.")

pd.DataFrame(
    hmap_mean,
    index=[f"dt={d}" for d in DELTA_TAU_2D],
    columns=[f"tw={t}" for t in TAU_WORLD_2D]
).to_csv(os.path.join(OUT, "heatmap_delta_tau_x_tau_world.csv"))


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2: Abruptness at Δτ=0 boundary
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 2: Abruptness at Δτ=0 boundary ...")

abrupt = {}
for dt in DELTA_TAU_FINE:
    vals = mean_A_self(seeds, dt)
    abrupt[dt] = (vals.mean(), vals.std())
    print(f"  Δτ={dt}  A={vals.mean():.4f}±{vals.std():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3: Noise robustness — σ_ξ sweep
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 3: σ_ξ robustness ...")

rob_xi = {}
for sx in SIGMA_XI_VALS:
    vals = mean_A_self(seeds, 1, sigma_xi=sx)
    rob_xi[sx] = (vals.mean(), vals.std())
    print(f"  σ_ξ={sx:.2f}  A={vals.mean():.4f}±{vals.std():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 4: Environmental noise robustness — σ_e sweep
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 4: σ_e robustness ...")

rob_e = {}
for se in SIGMA_E_VALS:
    vals = mean_A_self(seeds, 1, sigma_e=se)
    rob_e[se] = (vals.mean(), vals.std())
    print(f"  σ_e={se:.2f}  A={vals.mean():.4f}±{vals.std():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 5: Learning rate robustness — η sweep
# ═════════════════════════════════════════════════════════════════════════════
print("Analysis 5: η robustness ...")

rob_eta = {}
for et in ETA_VALS:
    vals = mean_A_self(seeds, 1, eta=et)
    rob_eta[et] = (vals.mean(), vals.std())
    print(f"  η={et:.3f}  A={vals.mean():.4f}±{vals.std():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure A: 2D heatmap + sign map
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# heatmap of mean A
ax = axes[0]
vmax = np.abs(hmap_mean).max()
im = ax.imshow(hmap_mean, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, origin="lower")
ax.set_xticks(range(len(TAU_WORLD_2D)))
ax.set_xticklabels([str(t) for t in TAU_WORLD_2D])
ax.set_yticks(range(len(DELTA_TAU_2D)))
ax.set_yticklabels([str(d) for d in DELTA_TAU_2D])
ax.set_xlabel("τ_world", fontsize=12)
ax.set_ylabel("Δτ  (τ_R − τ_L)", fontsize=12)
ax.set_title("Mean agency index ⟨A⟩", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="⟨A⟩")

# annotate τ_L and τ_L+1 columns
tau_L_col  = TAU_WORLD_2D.index(TAU_L) if TAU_L in TAU_WORLD_2D else None
tau_R1_col = TAU_WORLD_2D.index(TAU_L+1) if TAU_L+1 in TAU_WORLD_2D else None
if tau_L_col  is not None:
    ax.axvline(tau_L_col  - 0.5, color="yellow", lw=1.5, ls="--")
    ax.axvline(tau_L_col  + 0.5, color="yellow", lw=1.5, ls="--")
if tau_R1_col is not None:
    ax.axvline(tau_R1_col - 0.5, color="cyan", lw=1.5, ls=":")
    ax.axvline(tau_R1_col + 0.5, color="cyan", lw=1.5, ls=":")

# sign map (direction of agency)
ax = axes[1]
sign_colors = np.where(np.abs(hmap_mean) < 0.05, 0, hmap_sign)
im2 = ax.imshow(sign_colors, aspect="auto",
                cmap="coolwarm", vmin=-1.5, vmax=1.5, origin="lower")
ax.set_xticks(range(len(TAU_WORLD_2D)))
ax.set_xticklabels([str(t) for t in TAU_WORLD_2D])
ax.set_yticks(range(len(DELTA_TAU_2D)))
ax.set_yticklabels([str(d) for d in DELTA_TAU_2D])
ax.set_xlabel("τ_world", fontsize=12)
ax.set_ylabel("Δτ  (τ_R − τ_L)", fontsize=12)
ax.set_title("Agency direction  (+: self, −: env, 0: undefined)",
             fontsize=11, fontweight="bold")

# Δτ=0 row boundary
ax.axhline(0.5, color="k", lw=2.0, ls="-")
ax.text(-0.4, 0.0, "Δτ=0\n(A≡0)", ha="center", va="center",
        fontsize=8, color="k", fontweight="bold")

cbar2 = plt.colorbar(im2, ax=ax, ticks=[-1, 0, 1])
cbar2.set_ticklabels(["env dominant", "undefined", "self dominant"])

fig.suptitle("2D structure of agency attribution: Δτ × τ_world",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_2d_heatmap.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_2d_heatmap.png"), dpi=150, bbox_inches="tight")
print("Fig A saved.")


# ═════════════════════════════════════════════════════════════════════════════
# Figure B: Abruptness + 3 robustness panels
# ═════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle("Abruptness and robustness analyses", fontsize=12, fontweight="bold")

# ── Panel A: abruptness at Δτ boundary ──
ax = axes2[0, 0]
dts   = sorted(abrupt.keys())
a_m   = [abrupt[d][0] for d in dts]
a_s   = [abrupt[d][1] for d in dts]
colors = ["#d62728" if d == 0 else "#1f77b4" for d in dts]
ax.bar(dts, a_m, yerr=a_s, color=colors, capsize=4,
       edgecolor="k", linewidth=0.6, width=0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Δτ", fontsize=11)
ax.set_ylabel("⟨A⟩ (self interval)", fontsize=10)
ax.set_title("(A) Abrupt onset at Δτ = 0", fontsize=10, fontweight="bold")
ax.annotate("Δτ=0:\nA≡0", xy=(0, 0.01), ha="center", color="#d62728",
            fontsize=9, fontweight="bold")

# ── Panel B: σ_ξ robustness ──
ax = axes2[0, 1]
xs = sorted(rob_xi.keys())
ax.errorbar(xs, [rob_xi[x][0] for x in xs], yerr=[rob_xi[x][1] for x in xs],
            fmt="o-", color="#2ca02c", capsize=4, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel("Observation noise σ_ξ", fontsize=11)
ax.set_ylabel("⟨A⟩ (self interval)", fontsize=10)
ax.set_title("(B) Robustness: observation noise σ_ξ  (Δτ=1)", fontsize=10, fontweight="bold")
ax.set_xscale("log")

# ── Panel C: σ_e robustness ──
ax = axes2[1, 0]
xs = sorted(rob_e.keys())
ax.errorbar(xs, [rob_e[x][0] for x in xs], yerr=[rob_e[x][1] for x in xs],
            fmt="s-", color="#ff7f0e", capsize=4, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel("Environmental noise σ_e", fontsize=11)
ax.set_ylabel("⟨A⟩ (self interval)", fontsize=10)
ax.set_title("(C) Robustness: environmental noise σ_e  (Δτ=1)", fontsize=10, fontweight="bold")
ax.set_xscale("log")

# ── Panel D: η robustness ──
ax = axes2[1, 1]
xs = sorted(rob_eta.keys())
ax.errorbar(xs, [rob_eta[x][0] for x in xs], yerr=[rob_eta[x][1] for x in xs],
            fmt="^-", color="#9467bd", capsize=4, lw=1.5, ms=5)
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel("Learning rate η", fontsize=11)
ax.set_ylabel("⟨A⟩ (self interval)", fontsize=10)
ax.set_title("(D) Robustness: learning rate η  (Δτ=1)", fontsize=10, fontweight="bold")
ax.set_xscale("log")

plt.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_abruptness_robustness.pdf"), dpi=150, bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "fig_abruptness_robustness.png"), dpi=150, bbox_inches="tight")
print("Fig B saved.")


# ═════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ═════════════════════════════════════════════════════════════════════════════

rows = []
for sx in SIGMA_XI_VALS:
    rows.append(dict(param="sigma_xi", value=sx,
                     A_mean=rob_xi[sx][0], A_sd=rob_xi[sx][1]))
for se in SIGMA_E_VALS:
    rows.append(dict(param="sigma_e", value=se,
                     A_mean=rob_e[se][0], A_sd=rob_e[se][1]))
for et in ETA_VALS:
    rows.append(dict(param="eta", value=et,
                     A_mean=rob_eta[et][0], A_sd=rob_eta[et][1]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_robustness.csv"), index=False)

print("\n=== SUMMARY ===")
print(f"2D heatmap: {len(DELTA_TAU_2D)}×{len(TAU_WORLD_2D)} = {len(DELTA_TAU_2D)*len(TAU_WORLD_2D)} conditions")
print(f"Δτ=0 row: all |A| < 0.05 → ",
      all(abs(hmap_mean[0, j]) < 0.05 for j in range(len(TAU_WORLD_2D))))
print(f"τ_world=τ_L  col (Δτ>0): all A > 0 → ",
      all(hmap_mean[i, TAU_WORLD_2D.index(TAU_L)] > 0.4
          for i in range(1, len(DELTA_TAU_2D))))
print(f"τ_world=τ_L+1 col (Δτ>0): all A < 0 → ",
      all(hmap_mean[i, TAU_WORLD_2D.index(TAU_L+1)] < -0.4
          for i in range(1, len(DELTA_TAU_2D))))
print("Done.")
