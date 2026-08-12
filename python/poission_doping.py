"""
Poisson pick-up statistics for helium nanodroplets.

Plots P(k), the probability of a droplet capturing exactly k dopant molecules,
as a function of the partial pressure in the pick-up cell.

    lambda   = sigma_pu * (p_d / kB T) * L
    sigma_pu = 15.5 A^2 * <N_He>^(2/3)
    P(k)     = lambda^k exp(-lambda) / k!

Figure is sized to the text width of a thesis page and kept short in height.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B
from math import factorial

# ----------------------------------------------------------------------
# Parameters -- adjust to your apparatus
# ----------------------------------------------------------------------
N_HE = 3000          # mean number of He atoms per droplet
L_CELL = 0.076       # length of the doping region [m]
T_CELL = 300.0       # temperature of the dopant vapour [K]
K_VALUES = [1, 2, 3, 4]      # which k to draw
LAMBDA_MAX = 4.0             # right edge of the axis, in units of lambda

UNIT = "mbar"        # "mbar" or "Torr"
PA_PER_UNIT = {"mbar": 100.0, "Torr": 133.322}[UNIT]

TEXTWIDTH_IN = 6.3   # thesis text width [inch]  (16 cm; A4 minus 2.5 cm margins)
HEIGHT_IN = 2.3      # figure height [inch]

OUTFILE = f"pickup_poisson_N{N_HE}"

# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
def pickup_cross_section(n_he):
    """Geometric pick-up cross section in m^2."""
    return 15.5e-20 * n_he ** (2.0 / 3.0)


def lambda_of_pressure(p, n_he=N_HE, L=L_CELL, T=T_CELL):
    """Mean number of capture events for a dopant pressure in UNIT."""
    number_density = p * PA_PER_UNIT / (k_B * T)
    return pickup_cross_section(n_he) * number_density * L


def poisson(k, lam):
    return lam ** k * np.exp(-lam) / factorial(k)


slope = lambda_of_pressure(1.0)      # lambda is linear in p
p_unity = 1.0 / slope                # pressure at which lambda = 1
p = np.linspace(0.0, LAMBDA_MAX * p_unity, 800)
lam = lambda_of_pressure(p)

# scale the x axis by a power of ten, as in the reference figure
exponent = int(np.floor(np.log10(p[-1])))
scale = 10.0 ** exponent

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
})

fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, HEIGHT_IN))

colors = plt.cm.viridis(np.linspace(0.05, 0.8, len(K_VALUES)))
for color, k in zip(colors, K_VALUES):
    ax.plot(p / scale, poisson(k, lam), color=color, label=f"$k = {k}$")

# working point for single-dopant measurements
lam_work = 0.3
p_work = lam_work / slope
ax.axvline(p_work / scale, color="0.4", lw=0.7, ls="--")
ax.text(p_work / scale * 1.06, 0.39, rf"$\lambda = {lam_work}$",
        fontsize=8, color="0.3", va="top")

ax.set_xlabel(rf"doping cell pressure $p_\mathrm{{d}}$  "
              rf"($10^{{{exponent}}}\,\mathrm{{{UNIT}}}$)")
ax.set_ylabel("pickup probability")
ax.set_xlim(0, p[-1] / scale)
ax.set_ylim(0, 0.4)
ax.legend(ncol=len(K_VALUES), frameon=False, loc="upper right",
          columnspacing=1.2, handlelength=1.4)

# secondary axis showing lambda directly
sec = ax.secondary_xaxis(
    "top",
    functions=(lambda x: lambda_of_pressure(x * scale),
               lambda l: l / slope / scale),
)
sec.set_xlabel(r"$\lambda$")

ax.tick_params(direction="in", top=False, right=True)
fig.tight_layout(pad=0.3)

fig.savefig(f"{OUTFILE}.pdf")
fig.savefig(f"{OUTFILE}.png", dpi=300)

print(f"<N_He>        = {N_HE}")
print(f"sigma_pu      = {pickup_cross_section(N_HE) * 1e20:.0f} A^2")
print(f"lambda = 1 at = {p_unity:.3e} {UNIT}")
print(f"working point = {p_work:.3e} {UNIT}  (lambda = {lam_work})")
for k in K_VALUES:
    print(f"  P({k}) at working point = {poisson(k, lam_work):.3f}")