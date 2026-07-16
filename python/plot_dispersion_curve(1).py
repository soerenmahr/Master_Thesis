#!/usr/bin/env python3
"""Plot the measured superfluid-helium dispersion curve.

No command-line arguments are required. Change the values in the
CONFIGURATION section and start the file directly from your IDE/debugger.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# The data file is expected next to this Python script.
INPUT_FILE = Path(__file__).resolve().parent / "dispersion_curve.txt"

# Units used in the two columns of the text file.
Q_UNIT = "nm^-1"       # allowed: "nm^-1", "A^-1", "m^-1"
ENERGY_UNIT = "J"      # allowed: "J", "K", "meV"

# Plot settings.
DRAW_LANDAU_TANGENT = True
SHOW_PLOT = True

# Output files are written next to the script.
OUTPUT_PNG = Path(__file__).resolve().parent / "dispersion_curve.png"
OUTPUT_PDF = Path(__file__).resolve().parent / "dispersion_curve.pdf"

# Exact SI constants.
K_B = 1.380649e-23       # Boltzmann constant [J/K]
HBAR = 1.054571817e-34   # reduced Planck constant [J s]


def load_two_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns and sort the rows by q."""
    try:
        data = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
    except OSError as exc:
        raise RuntimeError(f"Could not open data file:\n{path}") from exc

    if data.size == 0:
        raise ValueError(f"No numeric data found in:\n{path}")

    if data.ndim == 1:
        if data.shape[0] < 2:
            raise ValueError("The input file must contain at least two columns.")
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError("The input file must contain at least two columns.")

    q = data[:, 0]
    energy = data[:, 1]

    # Remove rows containing NaN or infinity.
    valid = np.isfinite(q) & np.isfinite(energy)
    q = q[valid]
    energy = energy[valid]

    if q.size < 2:
        raise ValueError("Fewer than two valid data rows were found.")

    # Ensure that the line is drawn from low to high momentum.
    order = np.argsort(q)
    q = q[order]
    energy = energy[order]

    # Average repeated q values, if the file contains any.
    unique_q, inverse = np.unique(q, return_inverse=True)
    if unique_q.size != q.size:
        energy_sum = np.zeros_like(unique_q)
        counts = np.zeros_like(unique_q)
        np.add.at(energy_sum, inverse, energy)
        np.add.at(counts, inverse, 1)
        q = unique_q
        energy = energy_sum / counts

    return q, energy


def q_to_si(q: np.ndarray, unit: str) -> np.ndarray:
    """Convert wave number to m^-1."""
    factors = {
        "nm^-1": 1.0e9,
        "A^-1": 1.0e10,
        "m^-1": 1.0,
    }

    if unit not in factors:
        raise ValueError(f"Unknown Q_UNIT: {unit!r}")

    return q * factors[unit]


def energy_to_si(energy: np.ndarray, unit: str) -> np.ndarray:
    """Convert energy to joules."""
    factors = {
        "J": 1.0,
        "K": K_B,
        "meV": 1.602176634e-22,
    }

    if unit not in factors:
        raise ValueError(f"Unknown ENERGY_UNIT: {unit!r}")

    return energy * factors[unit]


def main() -> None:
    print(f"Reading data from: {INPUT_FILE}")

    q_raw, energy_raw = load_two_columns(INPUT_FILE)

    # Convert to SI for calculations.
    q_si = q_to_si(q_raw, Q_UNIT)
    energy_si = energy_to_si(energy_raw, ENERGY_UNIT)

    # Convert to the conventional plotting units for liquid helium.
    q_angstrom = q_si / 1.0e10
    energy_kelvin = energy_si / K_B

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # Plot the actual data. No spline and no manually invented anchor points.
    marker_spacing = max(q_angstrom.size // 80, 1)
    ax.plot(
        q_angstrom,
        energy_kelvin,
        linewidth=1.6,
        marker="o",
        markersize=2.8,
        markevery=marker_spacing,
        label="data",
    )

    if DRAW_LANDAU_TANGENT:
        valid_for_landau = (q_si > 0.0) & (energy_si >= 0.0)

        if np.any(valid_for_landau):
            valid_indices = np.flatnonzero(valid_for_landau)

            # For p = hbar*q, Landau's criterion is v_c = min[E(q)/(hbar*q)].
            velocities = (
                energy_si[valid_for_landau]
                / (HBAR * q_si[valid_for_landau])
            )

            minimum_position = int(np.argmin(velocities))
            tangent_index = valid_indices[minimum_position]
            critical_velocity = float(velocities[minimum_position])

            q_tangent = q_angstrom[tangent_index]
            energy_tangent = energy_kelvin[tangent_index]

            tangent_q = np.array([0.0, q_angstrom.max()])
            tangent_energy = (energy_tangent / q_tangent) * tangent_q

            ax.plot(
                tangent_q,
                tangent_energy,
                linestyle="--",
                linewidth=1.2,
                label=rf"Landau tangent: $v_c={critical_velocity:.1f}\,\mathrm{{m/s}}$",
            )
            ax.plot(q_tangent, energy_tangent, "o", markersize=5.0)

            print(f"Landau minimum at q = {q_tangent:.6g} A^-1")
            print(f"Energy there       = {energy_tangent:.6g} K")
            print(f"Critical velocity  = {critical_velocity:.6g} m/s")
        else:
            print("Landau tangent skipped: no valid q > 0 data points.")

    ax.set_xlabel(r"Wave number $q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"Excitation energy $\varepsilon(q)/k_B$ (K)")
    ax.set_title("Dispersion relation of superfluid helium")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
