#!/usr/bin/env python3
"""
Zeit-Frequenz-Spektrogramme der drei Zentrifugen-Varianten.

Konvention (wie in den Papern):
    phi(t) = phi_0 + w_0*t + beta*t^2   ->   w(t) = w_0 + 2*beta*t
    Die Steigung einer Arm-Linie im Zeit-Frequenz-Diagramm ist also 2*beta.

(a) Konventionelle Zentrifuge (CFG):
    Spektrum wird in der Fourier-Ebene halbiert; blauer Arm beta_+ > 0,
    roter Arm beta_- = -beta_+.  Beide Arme starten bei (t=0, w_0) mit
    maximaler Intensitaet (Schnitt durch das Gauss-Maximum) und klingen
    nur noch ab.  Frequenzdifferenz: dw(t) = 4*beta*t  ->  Omega(t) = 2*beta*t.

(b) Konstant-Frequenz-Zentrifuge (cfCFG):
    Identischer Chirp beta_0 in beiden Armen, Verzoegerung dt
    ->  Omega_cf = beta_0 * dt = const.

(c) Ultraslow-Zentrifuge (usCFG):
    beta_L = beta_R + d_beta  ->  Omega_us(t) ~ beta_0*dt + d_beta*t,
    Winkelbeschleunigung eps = d_beta / (2*pi).

Ausgabe: centrifuge_spectrograms.png / .pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# Stil
# ----------------------------------------------------------------------
DARK = False                     # True -> dunkler Hintergrund
LANG_X, LANG_Y = "Time", "Optical Frequency"

# Same font sizing as the other full-textwidth figures (e.g. cfg_wall_roton.pdf),
# so all thesis figures share one typographic scale.
FIGSIZE = (9.0, 3.8)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

if DARK:
    BG, FG, FRAME, ACCENT = "#2b3040", "#e8e8ee", "#b8bcc8", "#7fb3d5"
    TAILC, MARKEDGE = "#b8bcc8", "white"
else:
    BG, FG, FRAME, ACCENT = "white", "black", "#555555", "#1f4e79"
    TAILC, MARKEDGE = "#888888", "#222222"

# Sichtbares Spektrum: rot (niedrige Frequenz) -> violett (hohe Frequenz)
SPECTRAL = LinearSegmentedColormap.from_list("vis", [
    "#8d1b0b", "#c43c07", "#e07b0a", "#e8b400", "#c9cc1a",
    "#4caf28", "#17a58a", "#1f9fc4", "#2f77cf", "#4a54c8",
    "#7b46c9", "#a878dd", "#c9a6e8",
])

# ----------------------------------------------------------------------
# Geometrie-Parameter (alles in willkuerlichen Einheiten der Achsen)
# ----------------------------------------------------------------------
XLIM, YLIM = (0.0, 10.0), (0.0, 10.0)

SLOPE = 1.35        # Steigung dw/dt = 2*beta_0
F_SPAN = 3.6        # halbe dargestellte spektrale Breite eines Arms
WIDTH = 0.30        # halbe Banddicke (senkrecht zur Chirp-Linie)
TAPER = 2.1          # F_SPAN / sigma der Gauss-Einhuellenden
TAIL = 0.60         # Laenge der gestrichelten Verlaengerung (Zeiteinheiten)

DT = 3.20           # Delta t zwischen den Armen (Panels b, c)
DBETA = 0.50        # Steigungsdifferenz 2*d_beta (Panel c)


# ----------------------------------------------------------------------
# Zeichenroutinen
# ----------------------------------------------------------------------
def chirped_band(ax, t_c, f_c, slope, f_span=F_SPAN, width=WIDTH,
                 part="full", cnorm=None, taper=TAPER, tail=TAIL,
                 n_seg=400, cmap=SPECTRAL, zorder=3):
    """Zeichnet einen gechirpten Puls als spektral eingefaerbtes Band.

    t_c, f_c : Zentrum des Pulses (bei part='upper'/'lower' der Scheitel)
    slope    : dw/dt = 2*beta
    part     : 'full'  -> volle Gauss-Einhuellende
               'upper' -> nur blaue Haelfte (Maximum am Scheitel)
               'lower' -> nur rote Haelfte  (Maximum am Scheitel)
    cnorm    : (f_min, f_max) fuer die Farbskala
    """
    if part == "full":
        u = np.linspace(-f_span, f_span, n_seg + 1)
    elif part == "upper":
        u = np.linspace(0.0, f_span, n_seg + 1)
    elif part == "lower":
        u = np.linspace(-f_span, 0.0, n_seg + 1)
    else:
        raise ValueError(part)

    sigma = f_span / taper
    env = np.exp(-(u / sigma) ** 2)          # Amplituden-Einhuellende
    f = f_c + u
    t = t_c + u / slope

    # Einheitsnormale zur Chirp-Linie
    L = np.hypot(1.0 / slope, 1.0)
    nx, ny = -1.0 / L, (1.0 / slope) / L
    h = width * env

    xu, yu = t + nx * h, f + ny * h
    xl, yl = t - nx * h, f - ny * h

    verts = np.stack([
        np.column_stack([xu[:-1], yu[:-1]]),
        np.column_stack([xu[1:], yu[1:]]),
        np.column_stack([xl[1:], yl[1:]]),
        np.column_stack([xl[:-1], yl[:-1]]),
    ], axis=1)

    f_mid = 0.5 * (f[:-1] + f[1:])
    v0, v1 = cnorm if cnorm else (f.min(), f.max())
    colors = cmap(np.clip((f_mid - v0) / (v1 - v0), 0, 1))

    ax.add_collection(PolyCollection(verts, facecolors=colors,
                                     edgecolors=colors, linewidths=0.4,
                                     zorder=zorder))

    # gestrichelte Verlaengerung in Richtung wachsender Zeit
    if tail:
        i = int(np.argmax(t))
        p0 = np.array([t[i], f[i]])
        p1 = p0 + np.array([1.0, slope]) * tail   # entlang der Chirp-Linie
        ax.plot(*zip(p0, p1), ls=(0, (4, 4)), lw=1.1, color=TAILC,
                alpha=0.75, zorder=2)
        return p1
    return np.array([t[int(np.argmax(t))], f[int(np.argmax(t))]])


def diamond(ax, t, f, filled=False, size=55):
    ax.scatter([t], [f], marker="D", s=size, zorder=6,
               facecolors=("#e8402a" if filled else BG),
               edgecolors=MARKEDGE, linewidths=1.3)


def double_arrow(ax, p0, p1, label, offset=(0.0, 0.0), **kw):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=FG, lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=7)
    mid = (0.5 * (p0[0] + p1[0]) + offset[0], 0.5 * (p0[1] + p1[1]) + offset[1])
    ax.text(*mid, label, color=FG, ha="center", va="center",
            fontsize=14, zorder=8,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.5), **kw)


def style_axes(ax, label):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FRAME)
        s.set_linewidth(1.2)
    ax.set_facecolor(BG)
    ax.set_xlabel(LANG_X, color=ACCENT, labelpad=6)
    ax.text(0.05, 0.93, label, transform=ax.transAxes, color=FG,
            fontsize=15, fontweight="bold", va="top")


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def panel_a(ax):
    """Konventionelle Zentrifuge: beta_- = -beta_+, Spektrum halbiert."""
    t0, f0 = 2.6, 5.0
    cn = (f0 - F_SPAN, f0 + F_SPAN)          # gemeinsame Farbskala (ein Spektrum)

    e_up = chirped_band(ax, t0, f0, +SLOPE, part="upper", cnorm=cn)  # blauer Arm
    e_dn = chirped_band(ax, t0, f0, -SLOPE, part="lower", cnorm=cn)  # roter Arm

    ax.text(e_up[0] + 0.55, e_up[1] - 0.15, r"$+\beta$", color=FG,
            fontsize=14, ha="left", va="center")
    ax.text(e_dn[0] + 0.55, e_dn[1] + 0.15, r"$-\beta$", color=FG,
            fontsize=14, ha="left", va="center")
    diamond(ax, t0, f0, filled=False)

    # 2*Omega: vertikaler Abstand der beiden Arme, hier an einer Stelle
    # abseits vom gemeinsamen Ursprung ausgewertet, da der Abstand mit t waechst.
    t_a = t0 + 0.65 * (F_SPAN / SLOPE)
    f_hi = f0 + SLOPE * (t_a - t0)
    f_lo = f0 - SLOPE * (t_a - t0)
    double_arrow(ax, (t_a, f_lo + 0.35), (t_a, f_hi - 0.35), r"$2\Omega_{\mathrm{cfg}}$",
                 offset=(0.95, 0.0))

    ax.text(0.30, 0.05, r"$\Omega_{\mathrm{cfg}}(t)=2\beta t$", transform=ax.transAxes,
            color=FG, fontsize=12, ha="center")


def panel_b(ax):
    """cfCFG: gleicher Chirp, Verzoegerung dt -> Omega = beta_0*dt."""
    # Shift the pulse pair to the right so the full two-pulse structure,
    # including the dashed tails, is centered in the panel and not clipped.
    t1, f1 = 3.10, 4.6
    cn1 = (f1 - F_SPAN, f1 + F_SPAN)
    e1 = chirped_band(ax, t1, f1, SLOPE, cnorm=cn1)            # frueher Arm
    e2 = chirped_band(ax, t1 + DT, f1, SLOPE, cnorm=cn1)       # spaeterer Arm

    # Marker der Zentralfrequenzen (definieren dt)
    diamond(ax, t1, f1, filled=False)
    diamond(ax, t1 + DT, f1, filled=True)

    # Delta t: horizontaler Abstand der beiden Pulszentren (Rauten)
    double_arrow(ax, (t1 + 0.35, f1), (t1 + DT - 0.35, f1), r"$\Delta t$")

    ax.text(e1[0] - 0.15, e1[1] + 0.45, r"$\beta_0$", color=FG, fontsize=14,
            ha="center")
    ax.text(e2[0] - 0.15, e2[1] + 0.45, r"$\beta_0$", color=FG, fontsize=14,
            ha="center")
    ax.text(0.52, 0.05, r"$\Omega_{\mathrm{cfg}}=\beta_0\Delta t=$ const.",
            transform=ax.transAxes, color=FG, fontsize=12, ha="center")


def panel_c(ax):
    """usCFG: beta_L = beta_R + d_beta -> lineares Anwachsen von Omega."""
    # Beide Arme starten im selben Punkt (t_x, f_x): Omega(t_x) = 0,
    # die Frequenzdifferenz waechst danach linear an.
    s_L, s_R = SLOPE + DBETA, SLOPE
    t_x, f_x = 1.4, 0.7
    f_c = f_x + F_SPAN
    t_L, t_R = t_x + F_SPAN / s_L, t_x + F_SPAN / s_R
    cn1 = (f_x, f_x + 2 * F_SPAN)

    e1 = chirped_band(ax, t_L, f_c, s_L, cnorm=cn1)            # beta_L
    e2 = chirped_band(ax, t_R, f_c, s_R, cnorm=cn1)            # beta_R

    diamond(ax, t_L, f_c, filled=False)
    diamond(ax, t_R, f_c, filled=True)

    ax.text(e1[0] - 0.35, e1[1] + 0.4, r"$\beta_0+\Delta\beta$", color=FG,
            fontsize=14, ha="center")
    ax.text(e2[0] + 0.05, e2[1] + 0.4, r"$\beta_0$", color=FG, fontsize=14,
            ha="center")
    ax.text(0.52, 0.05, r"$\Omega_{\mathrm{cfg}}(t)=\Delta\beta\,t$",
            transform=ax.transAxes, color=FG, fontsize=12, ha="center")



# ----------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)

    for ax, lab, fn in zip(axes, ["(a)", "(b)", "(c)"],
                           [panel_a, panel_b, panel_c]):
        style_axes(ax, lab)
        fn(ax)

    axes[0].set_ylabel(LANG_Y, color=ACCENT, labelpad=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"centrifuge_spectrograms.{ext}", dpi=300,
                    facecolor=BG, bbox_inches="tight")
    print("gespeichert: centrifuge_spectrograms.png / .pdf")


if __name__ == "__main__":
    main()
