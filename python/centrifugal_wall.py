"""
Rotational ladder of CS2 and OCS in helium nanodroplets.

Left axis  : transition frequency of the J -> J+2 Raman transition,
             dE(J) = 4BJ + 6B - 8DJ^3 - 36DJ^2 - 60DJ - 36D
             This is the quantity the optical centrifuge must be resonant with
             (f_CFG = dE/2). Its maximum defines the "centrifugal wall".

Right axis : total rotational energy,
             E(J) = B J(J+1) - D [J(J+1)]^2
             This is the quantity to compare with the roton energy of ~6 cm^-1.

Effective B, D from Chatterley et al., PRL 125, 013001 (2020).
"""

import numpy as np
import matplotlib.pyplot as plt

E_ROTON = 179.9          # GHz  (6 cm^-1)
JMAX_PLOT = 18
STEP = 2                 # Raman selection rule: J -> J+2; only the even-J ladder is shown

# Same styling as the dispersion-curve / density-of-states plots. This figure
# spans the full text width, so it is wider than tall rather than square.
FIGSIZE = (9.0, 4.8)
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# CS2 (S=C=S, centrosymmetric, I=0 for 32S) only populates even J.
# OCS is not centrosymmetric and populates all J, but only the even-J ladder
# (0 -> 2 -> 4 -> ...) is shown here.
MOLECULES = {
    "OCS":    dict(B=2.180, D=0.0095, color="#7c5cd6", Jkick=5),
    "CS$_2$": dict(B=0.730, D=0.0012, color="#d62246", Jkick=10),
}


def dE(J, B, D):
    """Frequency of the J -> J+2 Raman transition."""
    return 4*B*J + 6*B - 8*D*J**3 - 36*D*J**2 - 60*D*J - 36*D


def E(J, B, D):
    """Total rotational energy."""
    x = J*(J + 1)
    return B*x - D*x**2


fig, axL = plt.subplots(figsize=FIGSIZE)
axR = axL.twinx()

for name, p in MOLECULES.items():
    J = np.arange(0, JMAX_PLOT + 1, STEP)
    f = dE(J, p["B"], p["D"])
    e = E(J, p["B"], p["D"])

    # ---- left axis: step plot, plateau from J to J+2 --------------------
    ax_J = np.append(J, J[-1] + STEP)
    ax_f = np.append(f, f[-1])
    axL.step(ax_J, ax_f, where="post", color=p["color"], lw=1.9, zorder=3)

    # centrifugal wall = maximum of the transition frequency
    iw = int(np.argmax(f))
    axL.plot(J[iw] + STEP/2, f[iw], "o", color=p["color"], ms=7,
             mec="white", mew=1.3, zorder=5)

    # ---- right axis: total rotational energy ----------------------------
    axR.plot(J, e, "--", color=p["color"], lw=1.3, alpha=0.75, zorder=2)
    axR.plot(J, e, "o", color="white", mec=p["color"], mew=1.3, ms=4.5, zorder=2)

    axL.plot([], [], color=p["color"], lw=1.9, label=name)

# ---- roton energy -----------------------------------------------------------
axR.hlines(E_ROTON, 8, JMAX_PLOT, color="0.25", ls=":", lw=1.4, zorder=1)
axR.text(JMAX_PLOT - 0.3, E_ROTON + 4,
         r"roton energy, $\approx 180$ GHz",
         ha="right", va="bottom", color="0.25")

# ---- cosmetics --------------------------------------------------------------
axL.set_xlabel(r"$J$")
axL.set_ylabel(r"$\Delta E_{J\rightarrow J+2}$  (GHz)")
axR.set_ylabel(r"$E(J)$  (GHz)")

axL.set_xlim(0, JMAX_PLOT)
axL.set_ylim(0, 47)
axR.set_ylim(0, 225)
axL.set_xticks(np.arange(0, JMAX_PLOT + 1, 2))

axL.plot([], [], "--o", color="0.45", mfc="white", ms=4.5, lw=1.3,
         label=r"$E(J)$, right axis")
axL.legend(loc="upper left")
axL.grid(alpha=0.22, lw=0.5)

fig.tight_layout()
fig.savefig("python/cfg_wall_roton.pdf")
fig.savefig("python/cfg_wall_roton.png", dpi=200)

# ---- console summary --------------------------------------------------------
for name, p in MOLECULES.items():
    J = np.arange(0, 40, STEP)
    f, e = dE(J, p["B"], p["D"]), E(J, p["B"], p["D"])
    iw, ie = int(np.argmax(f)), int(np.argmax(e))
    print(f"{name.replace('$_2$','2')}:")
    print(f"   wall        J = {J[iw]:2d} -> {J[iw]+2:2d}   dE = {f[iw]:6.2f} GHz   f_CFG = {f[iw]/2:5.2f} GHz"
          f"   E = {e[iw]:6.1f} GHz ({e[iw]/29.979:.2f} cm^-1, {100*e[iw]/E_ROTON:.0f}% of roton)")
    print(f"   max E(J)    J = {J[ie]:2d}   E  = {e[ie]:6.1f} GHz ({e[ie]/29.979:.2f} cm^-1,"
          f" {100*e[ie]/E_ROTON:.0f}% of roton)")
    print(f"   kick limit  J = {p['Jkick']:2d}   E  = {E(p['Jkick'],p['B'],p['D']):6.1f} GHz"
          f" ({E(p['Jkick'],p['B'],p['D'])/29.979:.2f} cm^-1)")
