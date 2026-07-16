from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


# ============================================================
# Hardcoded settings
# ============================================================

# The text file must be in the same folder as this script.
DATA_FILE = Path(__file__).resolve().parent / "dispersion_curve.txt"

# Input data:
#   first column:  q in nm^-1
#   second column: excitation energy in joules
NUMBER_OF_Q_INTERVALS = 200_000
NUMBER_OF_FREQUENCY_BINS = 350

SHOW_PLOTS = True
SAVE_PLOTS = True


# ============================================================
# Physical constants
# ============================================================

PLANCK_CONSTANT = 6.62607015e-34  # J s, exact


def load_dispersion_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the first two columns of the text file.

    Column 1: q in nm^-1
    Column 2: excitation energy E in J
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the data file:\n{file_path}\n\n"
            "Place dispersion_curve.txt in the same folder as this script."
        )

    data = np.loadtxt(file_path, comments="#")

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            "The text file must contain at least two columns:\n"
            "q [nm^-1]    energy [J]"
        )

    q_nm_inverse = data[:, 0]
    energy_joule = data[:, 1]

    # Remove NaN and infinite entries.
    valid = np.isfinite(q_nm_inverse) & np.isfinite(energy_joule)
    q_nm_inverse = q_nm_inverse[valid]
    energy_joule = energy_joule[valid]

    # Sort by increasing q.
    order = np.argsort(q_nm_inverse)
    q_nm_inverse = q_nm_inverse[order]
    energy_joule = energy_joule[order]

    # PCHIP requires unique q values.
    q_nm_inverse, unique_indices = np.unique(
        q_nm_inverse,
        return_index=True,
    )
    energy_joule = energy_joule[unique_indices]

    if len(q_nm_inverse) < 3:
        raise ValueError("At least three different q values are required.")

    return q_nm_inverse, energy_joule


def joule_to_ghz(energy_joule: np.ndarray) -> np.ndarray:
    """
    Convert excitation energy E to its equivalent ordinary frequency f:

        E = h f

    The returned frequency is in GHz.
    """
    return energy_joule / PLANCK_CONSTANT / 1.0e9


def calculate_density_of_states(
    q_nm_inverse: np.ndarray,
    frequency_ghz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the three-dimensional excitation density of states.

    For an isotropic dispersion,

        dN / V = q^2 dq / (2 pi^2).

    q is kept in nm^-1, so the shell-state density is in nm^-3.
    Binning by frequency in GHz gives a DOS in

        states / (nm^3 GHz).

    Energy/frequency binning automatically includes every q branch
    that reaches the same excitation frequency.
    """

    # Shape-preserving interpolation through the measured data.
    dispersion = PchipInterpolator(
        q_nm_inverse,
        frequency_ghz,
        extrapolate=False,
    )

    # Divide the measured q range into thin spherical shells.
    q_edges = np.linspace(
        q_nm_inverse.min(),
        q_nm_inverse.max(),
        NUMBER_OF_Q_INTERVALS + 1,
    )

    q_midpoints = 0.5 * (q_edges[:-1] + q_edges[1:])
    frequency_midpoints = dispersion(q_midpoints)

    # Exact state density represented by each q shell:
    #
    # integral[q_lower to q_upper] q^2 dq / (2*pi^2)
    # = (q_upper^3 - q_lower^3) / (6*pi^2)
    #
    # Because q is in nm^-1, the units are nm^-3.
    shell_weights = (
        q_edges[1:] ** 3 - q_edges[:-1] ** 3
    ) / (6.0 * np.pi**2)

    frequency_edges = np.linspace(
        frequency_midpoints.min(),
        frequency_midpoints.max(),
        NUMBER_OF_FREQUENCY_BINS + 1,
    )

    states_per_volume, _ = np.histogram(
        frequency_midpoints,
        bins=frequency_edges,
        weights=shell_weights,
    )

    frequency_bin_widths = np.diff(frequency_edges)

    # Units: states / (nm^3 GHz)
    dos_per_ghz = states_per_volume / frequency_bin_widths

    frequency_centres = 0.5 * (
        frequency_edges[:-1] + frequency_edges[1:]
    )

    return (
        frequency_centres,
        dos_per_ghz,
        q_midpoints,
        frequency_midpoints,
    )


def main() -> None:
    # --------------------------------------------------------
    # Load the raw q [nm^-1] and E [J] data
    # --------------------------------------------------------
    q_nm_inverse, energy_joule = load_dispersion_data(DATA_FILE)

    # Convert E to the equivalent ordinary frequency:
    #
    # f = E / h
    frequency_ghz = joule_to_ghz(energy_joule)

    # --------------------------------------------------------
    # Calculate the density of states
    # --------------------------------------------------------
    (
        dos_frequency_ghz,
        dos_per_ghz,
        q_dense,
        frequency_dense_ghz,
    ) = calculate_density_of_states(
        q_nm_inverse,
        frequency_ghz,
    )

    # --------------------------------------------------------
    # Check DOS normalization
    # --------------------------------------------------------
    frequency_bin_width = (
        dos_frequency_ghz[1] - dos_frequency_ghz[0]
    )
    integrated_dos = np.sum(dos_per_ghz * frequency_bin_width)

    expected_mode_density = (
        q_nm_inverse.max() ** 3 - q_nm_inverse.min() ** 3
    ) / (6.0 * np.pi**2)

    print(f"Data file: {DATA_FILE}")
    print(f"Number of measured points: {len(q_nm_inverse)}")
    print()
    print(
        "q range: "
        f"{q_nm_inverse.min():.6g} to "
        f"{q_nm_inverse.max():.6g} nm^-1"
    )
    print(
        "Equivalent frequency range: "
        f"{frequency_ghz.min():.6g} to "
        f"{frequency_ghz.max():.6g} GHz"
    )
    print()
    print(
        "Integral of calculated DOS: "
        f"{integrated_dos:.6g} nm^-3"
    )
    print(
        "Expected number of modes:    "
        f"{expected_mode_density:.6g} nm^-3"
    )

    # --------------------------------------------------------
    # Plot the dispersion relation in GHz
    # --------------------------------------------------------
    dispersion_figure, dispersion_axis = plt.subplots(
        figsize=(7.0, 4.8)
    )

    dispersion_axis.plot(
        q_nm_inverse,
        frequency_ghz,
        "o",
        markersize=3,
        label="measured data",
    )

    dispersion_axis.plot(
        q_dense,
        frequency_dense_ghz,
        linewidth=1.5,
        label="PCHIP interpolation",
    )

    dispersion_axis.set_xlabel(
        r"Wavevector $q$ [$\mathrm{nm}^{-1}$]"
    )
    dispersion_axis.set_ylabel(
        r"Equivalent excitation frequency $f=E/h$ [GHz]"
    )
    dispersion_axis.set_title(
        "Superfluid-helium dispersion relation"
    )
    dispersion_axis.grid(alpha=0.25)
    dispersion_axis.legend()
    dispersion_figure.tight_layout()

    # --------------------------------------------------------
    # Plot the DOS against excitation frequency
    # --------------------------------------------------------
    dos_figure, dos_axis = plt.subplots(figsize=(7.0, 4.8))

    dos_axis.plot(
        dos_frequency_ghz,
        dos_per_ghz,
        linewidth=1.5,
    )

    dos_axis.set_xlabel(
        r"Equivalent excitation frequency $f=E/h$ [GHz]"
    )
    dos_axis.set_ylabel(
        r"$g(f)/V$ "
        r"[$\mathrm{nm}^{-3}\,\mathrm{GHz}^{-1}$]"
    )
    dos_axis.set_title(
        "Three-dimensional excitation density of states"
    )
    dos_axis.grid(alpha=0.25)
    dos_figure.tight_layout()

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------
    if SAVE_PLOTS:
        dispersion_output = (
            DATA_FILE.parent / "helium_dispersion_GHz.png"
        )
        dos_output = (
            DATA_FILE.parent / "helium_density_of_states_GHz.png"
        )

        dispersion_figure.savefig(
            dispersion_output,
            dpi=250,
            bbox_inches="tight",
        )
        dos_figure.savefig(
            dos_output,
            dpi=250,
            bbox_inches="tight",
        )

        print()
        print(f"Saved dispersion plot to: {dispersion_output}")
        print(f"Saved DOS plot to:        {dos_output}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
