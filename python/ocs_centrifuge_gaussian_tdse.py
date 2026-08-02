#!/usr/bin/env python3
r"""
Full time-dependent Schrödinger-equation simulation of an optical centrifuge
acting on OCS, using a Gaussian intensity envelope and the simplest rigid-rotor model.

WHAT IS INCLUDED
----------------
1. A linear rigid rotor with only one molecular constant B:

       E_J / h = B J(J+1)

   Centrifugal distortion D is deliberately omitted.

2. The complete optical-cycle-averaged polarizability interaction in the
   symmetry sector connected to the initial state |J=0,M=0>:

       V(t) = -U(t) sin^2(theta) cos^2(phi - phi_d(t))

   with

       U(t) = Delta_alpha E_0(t)^2 / 4.

   The code does NOT restrict the dynamics to M=J and does NOT use a
   rotating-wave approximation. All matrix elements allowed by the rank-2
   interaction are retained:

       Delta J = 0, +/-2,       Delta M = 0, +/-2.

3. A rotating-frame transformation. This is an exact change of basis for the
   effective rigid-rotor Hamiltonian; it is not an approximation. In that
   frame, the Hamiltonian in frequency units is

       H_rot(t)/h = B J(J+1) - f_d(t) M - U(t)/h * Q_x,

   where

       Q_x = sin^2(theta) cos^2(phi).

4. A Gaussian *intensity* envelope with 320 ps FWHM, centered at t=0:

       U(t)/h = (U0/h) exp[-4 ln(2) (t/FWHM)^2].

   The linear centrifuge chirp is applied across the FWHM interval, also
   centered at t=0:

       t <= -FWHM/2:          f_d(t) = 0
       -FWHM/2 < t < FWHM/2: f_d(t) rises linearly from 0 to f_final
       t >= FWHM/2:           f_d(t) = f_final.

   The figures show 200 ps before and after the FWHM interval, i.e. the
   requested window from -360 ps to +360 ps. A Gaussian is still about 3.0%
   of its peak at these boundaries, so the TDSE is internally propagated
   farther into both tails. This avoids starting in a nonzero field and
   gives an almost field-free final state for the end-state map.

5. Many independent centrifuge calculations are propagated simultaneously.
   Every calculation has the same Gaussian envelope and chirp interval, but a
   different final centrifuge frequency. Therefore each case has a different
   constant acceleration during the 320 ps chirp.

OUTPUT
------
The script saves:

    ocs_gaussian_fan.png
        3-D fan: time, instantaneous centrifuge frequency, and <J>, drawn as
        one connected surface mesh across every final-frequency sweep.
        Surface color is the actual standard deviation sigma_J.

    ocs_gaussian_final_map.png
        Final field-free populations P_J versus final centrifuge frequency.

    ocs_gaussian_trajectory.png
        Time-resolved P_J(t) for the largest final frequency.

    ocs_gaussian.npz
        Numerical data used for the plots.

DEPENDENCIES
------------
    numpy
    sympy
    matplotlib

Run with:
    python ocs_centrifuge_gaussian_tdse.py
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time as wall_time

import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.wigner import wigner_3j


# =============================================================================
# 1. CONFIGURATION: CHANGE THE PHYSICAL AND NUMERICAL PARAMETERS HERE
# =============================================================================

# --- Molecular constant -------------------------------------------------------
# Ground-vibrational-state rotational constant of 16O-12C-32S.
# Only B is used. There is no centrifugal-distortion term in this model.
B_GHZ = 6.081492475

# --- Polarizability anisotropy ------------------------------------------------
# Delta_alpha = alpha_parallel - alpha_perpendicular.
# The value below is the measured static polarizability-volume anisotropy of
# OCS: 5.34 Angstrom^3. For a real optical experiment, Delta_alpha is in
# principle frequency dependent; using the static value for a far-off-resonant
# near-IR field is an approximation.
DELTA_ALPHA_VOLUME_A3 = 5.34

# --- Peak laser intensity -----------------------------------------------------
# This is the cycle-averaged optical intensity.
# 1e11 W/cm^2 is intentionally strong so that a 320 ps sweep can be close to
# adiabatic. Decrease it to resolve more isolated ladder-climbing steps;
# increase it if the molecule fails to follow the desired branch.
PEAK_INTENSITY_W_CM2 = 1.0e11

# --- Gaussian pulse timing ---------------------------------------------------
# This is the FWHM of the *intensity* envelope and therefore also of U(t).
# The envelope and the chirp are both centered at t=0.
GAUSSIAN_FWHM_PS = 320.0

# The requested plots show 200 ps before the leading half-maximum point and
# 200 ps after the trailing half-maximum point. With a 320 ps FWHM this means
# the visible plot window is -360 ... +360 ps.
PLOT_EXTRA_TIME_PS = 200.0

# A Gaussian never becomes exactly zero. If we initialized the molecule at the
# visible left edge (-360 ps), the interaction would already be about 3% of
# its maximum and the initial |0,0> state would be suddenly quenched into a
# strong field. To avoid that artifact, propagation begins earlier and ends
# later, at times where U/U0 equals this small number. The plots still show
# only the requested -360 ... +360 ps window.
PROPAGATION_EDGE_FRACTION = 1.0e-4

# --- Map of final centrifuge frequencies -------------------------------------
# OCS Raman resonances in the B-only model occur at
#     f_(J->J+2) = B(2J+3).
# A 0...120 GHz map therefore contains the transitions up to J=8 -> 10.
F_FINAL_MIN_GHZ = 0.0
F_FINAL_MAX_GHZ = 120.0
N_FINAL_FREQUENCIES = 200

# --- Truncated |J,M> basis ----------------------------------------------------
# J_MAX must be above the highest rotational state reached and high enough to
# contain field-induced dressing. The code prints the final population at the
# upper boundary so that truncation can be checked.
J_MAX = 16

# --- Initial state ------------------------------------------------------------
INITIAL_J = 0
INITIAL_M = 0

# --- Numerical time grid -----------------------------------------------------
# The TDSE is stored every SAVED_TIME_STEP_PS. Between stored points the
# split-operator propagator takes PROPAGATION_SUBSTEPS smaller unitary steps.
# The default internal step is 0.5 ps / 10 = 0.05 ps. Reduce it to check
# convergence if the final populations change visibly.
SAVED_TIME_STEP_PS = 0.5
PROPAGATION_SUBSTEPS = 10

# --- Plot/output controls -----------------------------------------------------
# The fan mesh connects every propagated final frequency into one surface.
# FAN_TIME_STRIDE only thins the time axis (to keep rendering fast); the
# frequency axis always keeps every requested sweep connected.
FAN_TIME_STRIDE = 4

# Before the chirp starts, every sweep's instantaneous drive frequency is
# exactly 0, so the mesh has zero width there and disappears. Purely for
# rendering, the fan gives each row a minimum visual y-floor of this
# fraction of its own final frequency, so the (physically identical)
# pre-ramp pendular dynamics are visible as a thin sheet instead of a
# degenerate line. It has no effect once the real chirp fraction exceeds it.
FAN_PRERAMP_VISUAL_FLOOR = 0.02
OUTPUT_PREFIX = "ocs_gaussian"
# Opens an interactive window (mouse-draggable 3-D view) for every figure
# after saving. Set back to False for headless/batch runs.
SHOW_PLOTS = True


# =============================================================================
# 2. PHYSICAL UNIT CONVERSION
# =============================================================================

# Exact SI constants used only to convert Delta_alpha and intensity into U/h.
EPSILON_0 = 8.8541878128e-12       # vacuum permittivity, F/m
C_LIGHT = 299_792_458.0            # speed of light, m/s
PLANCK_H = 6.62607015e-34          # Planck constant, J s


def interaction_frequency_ghz(
    delta_alpha_volume_a3: float,
    intensity_w_cm2: float,
) -> float:
    r"""Convert molecular anisotropy and laser intensity into U/h in GHz.

    Experimental molecular polarizabilities are often quoted as a
    "polarizability volume" in Angstrom^3. The corresponding SI polarizability
    is

        Delta_alpha_SI = 4 pi epsilon_0 Delta_alpha_volume.

    For a monochromatic electric field with peak amplitude E_0,

        I = (1/2) c epsilon_0 E_0^2,
        U = Delta_alpha_SI E_0^2 / 4.

    Combining both expressions gives the interaction energy U. Dividing by h
    converts it to a frequency, which is convenient because B is also in GHz.
    """
    delta_alpha_volume_m3 = delta_alpha_volume_a3 * 1.0e-30
    delta_alpha_si = 4.0 * np.pi * EPSILON_0 * delta_alpha_volume_m3
    intensity_w_m2 = intensity_w_cm2 * 1.0e4
    electric_field_squared = 2.0 * intensity_w_m2 / (C_LIGHT * EPSILON_0)
    interaction_joule = delta_alpha_si * electric_field_squared / 4.0
    return interaction_joule / PLANCK_H / 1.0e9


U0_GHZ = interaction_frequency_ghz(
    DELTA_ALPHA_VOLUME_A3,
    PEAK_INTENSITY_W_CM2,
)


# =============================================================================
# 3. BUILD THE ROTATIONAL BASIS |J,M>
# =============================================================================


@dataclass(frozen=True)
class RotorBasis:
    """Container holding the truncated |J,M> basis and lookup arrays."""

    states: tuple[tuple[int, int], ...]
    index: dict[tuple[int, int], int]
    J: np.ndarray
    M: np.ndarray


def build_basis(j_max: int) -> RotorBasis:
    r"""Build the complete invariant sector connected to |0,0>.

    The rank-2 polarizability interaction preserves the parity of J and M.
    Starting from |0,0>, only even J and even M are reachable. Restricting to
    these states is therefore an exact symmetry reduction, not an RWA.
    """
    states: list[tuple[int, int]] = []

    # J=0,2,4,... because INITIAL_J is even and Delta J is 0 or +/-2.
    for J in range(0, j_max + 1, 2):
        # For each J, include every physically allowed even M from -J to +J.
        for M in range(-J, J + 1, 2):
            states.append((J, M))

    index = {state: i for i, state in enumerate(states)}
    j_array = np.array([state[0] for state in states], dtype=float)
    m_array = np.array([state[1] for state in states], dtype=float)
    return RotorBasis(tuple(states), index, j_array, m_array)


# =============================================================================
# 4. BUILD Q_x = sin^2(theta) cos^2(phi) FROM WIGNER 3-j SYMBOLS
# =============================================================================


@lru_cache(maxsize=None)
def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Cached floating-point Wigner 3-j symbol."""
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def ylm_matrix_element(
    J_bra: int,
    M_bra: int,
    k: int,
    q: int,
    J_ket: int,
    M_ket: int,
) -> float:
    r"""Calculate <J_bra,M_bra|Y_k^q|J_ket,M_ket>.

    The two Wigner 3-j symbols automatically impose the angular-momentum
    selection rules. For the present k=2 interaction, only Delta J=0,+/-2 and
    Delta M=q=0,+/-2 survive.
    """
    prefactor = ((-1.0) ** M_bra) * np.sqrt(
        (2 * J_bra + 1)
        * (2 * k + 1)
        * (2 * J_ket + 1)
        / (4.0 * np.pi)
    )
    return prefactor * w3j(J_bra, k, J_ket, 0, 0, 0) * w3j(
        J_bra, k, J_ket, -M_bra, q, M_ket
    )


def build_qx_matrix(basis: RotorBasis) -> np.ndarray:
    r"""Construct the full matrix of Q_x=sin^2(theta)cos^2(phi).

    The spherical-harmonic expansion is

        Q_x = 1/3
              - sqrt(4 pi / 45) Y_2^0
              + sqrt(2 pi / 15) [Y_2^2 + Y_2^-2].

    The q=0 term generates Delta M=0 couplings. The q=+/-2 terms generate
    Delta M=+/-2 couplings. Nothing is manually restricted to the M=J chain.
    """
    n_states = len(basis.states)

    # The constant 1/3 term contributes 1/3 to every diagonal element.
    qx = np.eye(n_states, dtype=np.complex128) / 3.0

    harmonic_coefficients = {
        0: -np.sqrt(4.0 * np.pi / 45.0),
        +2: np.sqrt(2.0 * np.pi / 15.0),
        -2: np.sqrt(2.0 * np.pi / 15.0),
    }

    # Fill the matrix column by column, where each column is a ket |J,M>.
    for column, (J_ket, M_ket) in enumerate(basis.states):
        for q, coefficient in harmonic_coefficients.items():
            M_bra = M_ket + q

            # A rank-2 operator only connects J to J-2, J, and J+2.
            for J_bra in (J_ket - 2, J_ket, J_ket + 2):
                if J_bra < 0 or J_bra > J_MAX:
                    continue

                row = basis.index.get((J_bra, M_bra))
                if row is None:
                    continue

                qx[row, column] += coefficient * ylm_matrix_element(
                    J_bra,
                    M_bra,
                    2,
                    q,
                    J_ket,
                    M_ket,
                )

    # Floating conversion of exact Wigner symbols may create roundoff-level
    # asymmetry. Symmetrization restores an exactly Hermitian numerical matrix.
    qx = 0.5 * (qx + qx.conj().T)

    error = np.max(np.abs(qx - qx.conj().T))
    if error > 1.0e-12:
        raise RuntimeError(f"Q_x matrix is not Hermitian: max error={error:g}")

    return qx


# =============================================================================
# 5. DEFINE THE GAUSSIAN ENVELOPE AND THE LINEAR CENTRIFUGE CHIRP
# =============================================================================


def gaussian_interaction_ghz(t_ps: float | np.ndarray) -> np.ndarray:
    r"""Return the optical-cycle-averaged interaction U(t)/h in GHz.

    The laser *intensity* is Gaussian with the configured FWHM. Since the
    polarizability interaction is proportional to intensity, U(t) has the same
    FWHM:

        U(t)/h = (U0/h) exp[-4 ln(2) (t/FWHM)^2].

    The pulse center is t=0. At t=+/-FWHM/2 the interaction is exactly U0/2.
    """
    t = np.asarray(t_ps, dtype=float)
    return U0_GHZ * np.exp(
        -4.0 * np.log(2.0) * (t / GAUSSIAN_FWHM_PS) ** 2
    )


def chirp_fraction(t_ps: float | np.ndarray) -> np.ndarray:
    r"""Return the fraction of the requested final rotation frequency.

    The frequency is held at zero before the leading half-maximum point,
    increases linearly across the FWHM interval, and is held at the
    requested final value after the trailing half-maximum point.

    For a calculation with final frequency f_final,

        f_d(t) = f_final * chirp_fraction(t).

    Thus every run has a constant acceleration during the central interval,
    but a different acceleration because f_final is different.
    """
    t = np.asarray(t_ps, dtype=float)
    half_fwhm = 0.5 * GAUSSIAN_FWHM_PS

    fraction = np.zeros_like(t)

    # Linear ramp from 0 to 1 between -FWHM/2 and +FWHM/2.
    in_chirp = (t > -half_fwhm) & (t < half_fwhm)
    fraction[in_chirp] = (
        t[in_chirp] + half_fwhm
    ) / GAUSSIAN_FWHM_PS

    # After the chirp interval the polarization keeps rotating at f_final while
    # the Gaussian tail removes the field-induced dressing.
    fraction[t >= half_fwhm] = 1.0
    return fraction


def pulse_schedule(t_ps: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Return both time-dependent pulse quantities used by the Hamiltonian.

    Returns
    -------
    sweep_fraction:
        Dimensionless number between 0 and 1. Multiplication by each run's
        f_final gives its instantaneous centrifuge rotation frequency.

    interaction_ghz:
        The Gaussian polarizability interaction U(t)/h in GHz.
    """
    return chirp_fraction(t_ps), gaussian_interaction_ghz(t_ps)


def gaussian_cutoff_time_ps(edge_fraction: float) -> float:
    r"""Find |t| where the Gaussian interaction has fallen to edge_fraction.

    Solving

        exp[-4 ln(2) (t/FWHM)^2] = edge_fraction

    gives the propagation half-width used to start and end the TDSE in an
    almost field-free region.
    """
    if not (0.0 < edge_fraction < 1.0):
        raise ValueError("PROPAGATION_EDGE_FRACTION must lie between 0 and 1")
    return GAUSSIAN_FWHM_PS * np.sqrt(
        -np.log(edge_fraction) / (4.0 * np.log(2.0))
    )


# Visible interval requested by the user: PLOT_EXTRA_TIME_PS outside each
# FWHM edge, centered on t=0.
PLOT_TIME_MIN_PS = -0.5 * GAUSSIAN_FWHM_PS - PLOT_EXTRA_TIME_PS
PLOT_TIME_MAX_PS = +0.5 * GAUSSIAN_FWHM_PS + PLOT_EXTRA_TIME_PS

# Longer internal interval used to start from |0,0> in an almost vanishing
# field and to let the final dressed state undress before P_J is reported.
PROPAGATION_HALF_WIDTH_PS = gaussian_cutoff_time_ps(
    PROPAGATION_EDGE_FRACTION
)
PROPAGATION_TIME_MIN_PS = -PROPAGATION_HALF_WIDTH_PS
PROPAGATION_TIME_MAX_PS = +PROPAGATION_HALF_WIDTH_PS

# =============================================================================
# 6. PROPAGATE THE FULL TDSE
# =============================================================================


@dataclass
class SimulationResult:
    """All observables saved for one final centrifuge frequency."""

    final_frequency_ghz: float
    time_ps: np.ndarray
    drive_frequency_ghz: np.ndarray
    interaction_ghz: np.ndarray
    psi_t: np.ndarray
    p_j_t: np.ndarray
    j_values: np.ndarray
    mean_j: np.ndarray
    sigma_j: np.ndarray
    norm_error: float

    @property
    def final_p_j(self) -> np.ndarray:
        return self.p_j_t[:, -1]

    @property
    def final_mean_j(self) -> float:
        return float(self.mean_j[-1])

    @property
    def final_dominant_j(self) -> int:
        return int(self.j_values[np.argmax(self.final_p_j)])


class CentrifugeSimulator:
    """Precompute static matrices and propagate many centrifuges at once."""

    def __init__(self, basis: RotorBasis, qx: np.ndarray):
        self.basis = basis
        self.qx = qx

        # Field-free rigid-rotor energies in frequency units. This is the only
        # molecular energy model used: E_J/h = B J(J+1).
        self.field_free_ghz = B_GHZ * basis.J * (basis.J + 1.0)

        # List of distinct J values and masks used later to sum over M:
        #     P_J(t) = sum_M |c_(J,M)(t)|^2.
        self.j_values = np.array(sorted({J for J, _ in basis.states}), dtype=int)
        self.j_masks = np.array(
            [
                [1.0 if state[0] == J else 0.0 for state in basis.states]
                for J in self.j_values
            ]
        )

        # Diagonalize Q_x once. The interaction propagator can then be applied
        # exactly at every small time step without repeatedly diagonalizing it.
        self.q_eigenvalues, self.q_eigenvectors = np.linalg.eigh(self.qx)
        self.q_eigenvectors_h = self.q_eigenvectors.conj().T

    def simulate_many(self, final_frequencies_ghz: np.ndarray) -> list[SimulationResult]:
        r"""Solve the TDSE for all requested final frequencies.

        Each column of psi is one independent wave function. Propagating all
        columns together is mathematically identical to separate calculations,
        but much faster because NumPy can use matrix-matrix multiplication.

        The short-time propagator uses symmetric Strang splitting:

          exp[-i H dt/hbar]
            ~= exp[-i H_diag dt/(2 hbar)]
               exp[-i H_int  dt/hbar]
               exp[-i H_diag dt/(2 hbar)].

        Here H_diag contains the rigid-rotor energy and the rotating-frame term
        -h f_d M. H_int=-h U Q_x. Every factor is unitary, so norm conservation
        is excellent.
        """
        final_frequencies = np.asarray(final_frequencies_ghz, dtype=float)
        if final_frequencies.ndim != 1 or len(final_frequencies) == 0:
            raise ValueError("final_frequencies_ghz must be a nonempty 1-D array")

        # The TDSE is propagated over the long Gaussian tails, not merely over
        # the visible plotting window. This makes the initial and final states
        # almost field free. The number of saved samples is derived from the
        # requested saved-time spacing.
        propagation_duration_ps = (
            PROPAGATION_TIME_MAX_PS - PROPAGATION_TIME_MIN_PS
        )
        n_time_samples = int(
            np.ceil(propagation_duration_ps / SAVED_TIME_STEP_PS)
        ) + 1
        time_ps = np.linspace(
            PROPAGATION_TIME_MIN_PS,
            PROPAGATION_TIME_MAX_PS,
            n_time_samples,
        )

        n_states = len(self.basis.states)
        n_cases = len(final_frequencies)

        # Initial condition: every simulation begins in the same field-free
        # rotational ground state |0,0>.
        psi = np.zeros((n_states, n_cases), dtype=np.complex128)
        initial_index = self.basis.index.get((INITIAL_J, INITIAL_M))
        if initial_index is None:
            raise ValueError("Initial state is not contained in the selected basis")
        psi[initial_index, :] = 1.0

        # Save the complete wave function at every requested output time.
        psi_all_t = np.empty(
            (n_states, n_cases, n_time_samples),
            dtype=np.complex128,
        )
        psi_all_t[:, :, 0] = psi

        saved_dt_ps = time_ps[1] - time_ps[0]
        internal_dt_ps = saved_dt_ps / PROPAGATION_SUBSTEPS

        # GHz * ps = 10^-3. Therefore the phase generated by H/h in GHz is
        #     exp[-i 2 pi * frequency_GHz * dt_ps * 10^-3].
        ghz_ps_to_phase = 2.0 * np.pi * 1.0e-3

        current_time_ps = float(time_ps[0])

        # Advance from one saved output time to the next.
        for sample_index in range(1, n_time_samples):
            for _ in range(PROPAGATION_SUBSTEPS):
                midpoint_ps = current_time_ps + 0.5 * internal_dt_ps
                fraction_array, interaction_array = pulse_schedule(midpoint_ps)
                sweep_fraction = float(fraction_array)
                interaction_ghz = float(interaction_array)

                # Each final-frequency case has its own instantaneous f_d(t).
                drive_frequencies_ghz = final_frequencies * sweep_fraction

                # Diagonal rotating-frame Hamiltonian for every basis state and
                # every independent final-frequency case:
                #     H_diag/h = B J(J+1) - f_d(t) M.
                diagonal_ghz = (
                    self.field_free_ghz[:, None]
                    - self.basis.M[:, None] * drive_frequencies_ghz[None, :]
                )

                # First half-step under H_diag.
                half_phase = np.exp(
                    -0.5j
                    * ghz_ps_to_phase
                    * internal_dt_ps
                    * diagonal_ghz
                )
                psi *= half_phase

                # Full step under H_int/h = -U(t) Q_x.
                # In the Q_x eigenbasis this operator is diagonal. The minus
                # sign in H_int produces the plus sign in the exponential.
                psi_in_q_basis = self.q_eigenvectors_h @ psi
                interaction_phase = np.exp(
                    +1j
                    * ghz_ps_to_phase
                    * internal_dt_ps
                    * interaction_ghz
                    * self.q_eigenvalues
                )
                psi_in_q_basis *= interaction_phase[:, None]
                psi = self.q_eigenvectors @ psi_in_q_basis

                # Second half-step under H_diag completes the symmetric split.
                psi *= half_phase
                current_time_ps += internal_dt_ps

            # Renormalize only to suppress accumulated floating-point roundoff.
            # The split propagator itself is unitary.
            norms = np.linalg.norm(psi, axis=0)
            psi /= norms[None, :]
            psi_all_t[:, :, sample_index] = psi

        # The pulse shape is common to all cases, except that the frequency is
        # multiplied by the case-specific f_final.
        sweep_fraction_t, interaction_t = pulse_schedule(time_ps)

        results: list[SimulationResult] = []

        # Convert each propagated wave function into J-resolved observables.
        for case_index, final_frequency in enumerate(final_frequencies):
            psi_t = psi_all_t[:, case_index, :]
            state_probabilities = np.abs(psi_t) ** 2

            # Sum over M to get the probability of each rotational J manifold.
            p_j_t = self.j_masks @ state_probabilities

            # Check norm conservation at all saved times.
            total_probability = np.sum(state_probabilities, axis=0)
            norm_error = float(np.max(np.abs(total_probability - 1.0)))

            # Expectation value and standard deviation of J.
            mean_j = self.j_values @ p_j_t
            mean_j_squared = (self.j_values.astype(float) ** 2) @ p_j_t
            sigma_j = np.sqrt(np.maximum(mean_j_squared - mean_j**2, 0.0))

            results.append(
                SimulationResult(
                    final_frequency_ghz=float(final_frequency),
                    time_ps=time_ps,
                    drive_frequency_ghz=final_frequency * sweep_fraction_t,
                    interaction_ghz=interaction_t,
                    psi_t=psi_t,
                    p_j_t=p_j_t,
                    j_values=self.j_values,
                    mean_j=mean_j,
                    sigma_j=sigma_j,
                    norm_error=norm_error,
                )
            )

        return results


# =============================================================================
# 7. ANALYTIC RIGID-ROTOR RESONANCE POSITIONS FOR REFERENCE
# =============================================================================


def ideal_raman_resonances(j_max: int) -> list[tuple[int, float]]:
    r"""Return B-only resonances for |J,J> -> |J+2,J+2>.

    The rotating field transfers Delta M=+2, so resonance occurs when

        E_(J+2)-E_J = 2 h f_d.

    With E_J/h=BJ(J+1),

        f_d = B(2J+3).
    """
    return [(J, B_GHZ * (2 * J + 3)) for J in range(0, j_max - 1, 2)]


# =============================================================================
# 8. PLOTS
# =============================================================================


def plot_final_map(results: list[SimulationResult], output: Path) -> None:
    """Plot final field-free P_J for every final centrifuge frequency."""
    final_frequencies = np.array([r.final_frequency_ghz for r in results])
    j_values = results[0].j_values
    final_population_map = np.column_stack([r.final_p_j for r in results])
    final_mean_j = np.array([r.final_mean_j for r in results])
    final_dominant_j = np.array([r.final_dominant_j for r in results])

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    mesh = ax.pcolormesh(
        final_frequencies,
        j_values,
        final_population_map,
        shading="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax, label=r"final field-free population $P_J$")

    ax.plot(final_frequencies, final_mean_j, lw=1.8, label=r"final $\langle J\rangle$")
    ax.step(
        final_frequencies,
        final_dominant_j,
        where="mid",
        lw=1.3,
        ls="--",
        label="dominant final J",
    )

    # Add the uncoupled B-only resonance positions as vertical reference lines.
    for J, resonance_ghz in ideal_raman_resonances(J_MAX):
        if final_frequencies.min() <= resonance_ghz <= final_frequencies.max():
            ax.axvline(resonance_ghz, lw=0.7, alpha=0.35)
            ax.text(
                resonance_ghz,
                j_values[-1] + 0.2,
                f"{J}→{J+2}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("final centrifuge rotation frequency (GHz)")
    ax.set_ylabel(r"field-free rotational quantum number $J$")
    ax.set_title("OCS rigid rotor: field-free populations after the Gaussian tail")
    ax.set_yticks(j_values)
    ax.set_ylim(j_values[0] - 0.5, j_values[-1] + 1.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_single_trajectory(result: SimulationResult, output: Path) -> None:
    """Plot P_J(t) for the simulation with the largest final frequency."""
    fig, ax = plt.subplots(figsize=(9.2, 5.7))

    mesh = ax.pcolormesh(
        result.time_ps,
        result.j_values,
        result.p_j_t,
        shading="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax, label=r"population $P_J(t)$")

    ax.plot(result.time_ps, result.mean_j, lw=1.7, label=r"$\langle J\rangle$")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"$J$")
    ax.set_yticks(result.j_values)
    ax.set_title(
        f"OCS full TDSE: final centrifuge frequency "
        f"{result.final_frequency_ghz:.1f} GHz"
    )

    # Show exactly the requested interval: PLOT_EXTRA_TIME_PS before and after
    # the FWHM interval. The TDSE itself was propagated farther into the tails.
    ax.set_xlim(PLOT_TIME_MIN_PS, PLOT_TIME_MAX_PS)
    half_fwhm = 0.5 * GAUSSIAN_FWHM_PS
    ax.axvline(-half_fwhm, lw=0.8, ls=":", alpha=0.5)
    ax.axvline(+half_fwhm, lw=0.8, ls=":", alpha=0.5)

    # Add the drive frequency and a rescaled intensity envelope on a second axis.
    ax2 = ax.twinx()
    ax2.plot(
        result.time_ps,
        result.drive_frequency_ghz,
        lw=1.0,
        ls="--",
        label=r"$f_d(t)$",
    )
    if np.max(result.interaction_ghz) > 0.0:
        scaled_interaction = result.interaction_ghz / np.max(result.interaction_ghz)
        ax2.plot(
            result.time_ps,
            scaled_interaction * result.final_frequency_ghz,
            lw=0.9,
            ls=":",
            label="scaled intensity envelope",
        )
    ax2.set_ylabel("drive frequency (GHz)")

    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_fan(results: list[SimulationResult], output: Path) -> None:
    r"""Plot every propagated final frequency as one connected 3-D surface.

    Rather than drawing a handful of representative rays, this builds a
    single mesh:
      x = physical pulse time,
      y = instantaneous centrifuge frequency,
      z = actual TDSE expectation value <J>.

    Every result shares the same time grid, so neighboring final
    frequencies at neighboring times form genuine mesh quads -- all
    requested sweeps are connected into one surface, not drawn as
    independent lines. The surface is colored by the actual standard
    deviation sigma_J of the instantaneous state, which previously was
    shown as a ribbon around each ray.

    Before the chirp starts, the true instantaneous drive frequency is 0 for
    every sweep, so the mesh would otherwise collapse to a zero-width line
    there (all rows genuinely share the same dynamics pre-ramp). A small
    per-row visual floor (FAN_PRERAMP_VISUAL_FLOOR) is added to y so that
    interval is still visible as a thin sheet; it never changes z, and it
    fades out the instant the real chirp fraction takes over.
    """
    fig = plt.figure(figsize=(10.2, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    # The full propagation extends farther into the Gaussian tails. For the
    # fan, keep only the user-requested display window. All
    # results share the same time grid, so this mask applies to every case.
    visible = (
        (results[0].time_ps >= PLOT_TIME_MIN_PS)
        & (results[0].time_ps <= PLOT_TIME_MAX_PS)
    )
    time_visible = results[0].time_ps[visible]

    n_cases = len(results)
    n_time = int(np.sum(visible))

    # The true sweep fraction is common to every case; only its scale (each
    # case's final frequency) differs. Flooring it before scaling keeps the
    # visual offset proportional to each row's own final frequency.
    visual_fraction = np.maximum(
        chirp_fraction(time_visible), FAN_PRERAMP_VISUAL_FLOOR
    )

    x = np.tile(time_visible, (n_cases, 1))
    y = np.empty((n_cases, n_time))
    z = np.empty((n_cases, n_time))
    sigma = np.empty((n_cases, n_time))
    for row, result in enumerate(results):
        y[row] = result.final_frequency_ghz * visual_fraction
        z[row] = result.mean_j[visible]
        sigma[row] = result.sigma_j[visible]

    norm = plt.Normalize(vmin=float(sigma.min()), vmax=float(sigma.max()))
    face_colors = plt.cm.viridis(norm(sigma))

    # rstride=1 keeps every final-frequency sweep connected to its neighbors.
    # cstride only thins the time axis, purely to keep the polygon count
    # (and therefore render time) manageable.
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=FAN_TIME_STRIDE,
        facecolors=face_colors,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(sigma)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=r"$\sigma_J$")

    ax.set_xlabel("pulse time (ps)")
    ax.set_ylabel("instantaneous centrifuge frequency (GHz)")
    ax.set_zlabel(r"$\langle J\rangle$")
    ax.set_title(f"OCS optical centrifuge: Gaussian FWHM {GAUSSIAN_FWHM_PS:.0f} ps")
    ax.view_init(elev=21, azim=-127)
    ax.set_box_aspect((1.0, 1.05, 0.75))

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    if not SHOW_PLOTS:
        plt.close(fig)


# =============================================================================
# 9. SAVE NUMERICAL DATA
# =============================================================================


def save_results(results: list[SimulationResult], output: Path) -> None:
    """Store the most useful arrays in a compressed NumPy archive."""
    np.savez_compressed(
        output,
        B_GHZ=B_GHZ,
        delta_alpha_volume_A3=DELTA_ALPHA_VOLUME_A3,
        peak_intensity_W_cm2=PEAK_INTENSITY_W_CM2,
        peak_interaction_GHz=U0_GHZ,
        gaussian_FWHM_ps=GAUSSIAN_FWHM_PS,
        plot_extra_time_ps=PLOT_EXTRA_TIME_PS,
        propagation_edge_fraction=PROPAGATION_EDGE_FRACTION,
        propagation_time_min_ps=PROPAGATION_TIME_MIN_PS,
        propagation_time_max_ps=PROPAGATION_TIME_MAX_PS,
        final_frequencies_GHz=np.array(
            [r.final_frequency_ghz for r in results]
        ),
        J_values=results[0].j_values,
        final_P_J=np.column_stack([r.final_p_j for r in results]),
        final_mean_J=np.array([r.final_mean_j for r in results]),
        final_dominant_J=np.array([r.final_dominant_j for r in results]),
        time_ps=results[-1].time_ps,
        largest_frequency_P_J_t=results[-1].p_j_t,
        largest_frequency_mean_J=results[-1].mean_j,
        largest_frequency_drive_GHz=results[-1].drive_frequency_ghz,
        common_interaction_GHz=results[-1].interaction_ghz,
    )


# =============================================================================
# 10. MAIN PROGRAM: EXECUTE THE STEPS IN ORDER
# =============================================================================


def main() -> None:
    start_time = wall_time.perf_counter()
    output_directory = Path.cwd()

    # Step 1: report the physical model and the intensity-to-coupling conversion.
    print("OCS optical-centrifuge TDSE simulation")
    print("---------------------------------------")
    print(f"B                         = {B_GHZ:.9f} GHz")
    print(f"Delta alpha               = {DELTA_ALPHA_VOLUME_A3:.3f} Angstrom^3")
    print(f"peak intensity            = {PEAK_INTENSITY_W_CM2:.3e} W/cm^2")
    print(f"peak interaction U0/h     = {U0_GHZ:.3f} GHz")
    print(f"dimensionless U0/B        = {U0_GHZ / B_GHZ:.3f}")
    visible_edge_fraction = float(
        gaussian_interaction_ghz(PLOT_TIME_MAX_PS) / U0_GHZ
    )
    print(f"Gaussian intensity FWHM   = {GAUSSIAN_FWHM_PS:.1f} ps")
    print(
        f"visible plot interval     = {PLOT_TIME_MIN_PS:.1f} ... "
        f"{PLOT_TIME_MAX_PS:.1f} ps"
    )
    print(
        f"U/U0 at visible edges     = {visible_edge_fraction:.4f} "
        "(still dressed there)"
    )
    print(
        f"TDSE propagation interval = {PROPAGATION_TIME_MIN_PS:.1f} ... "
        f"{PROPAGATION_TIME_MAX_PS:.1f} ps"
    )
    print(
        f"U/U0 at TDSE boundaries   = {PROPAGATION_EDGE_FRACTION:.1e}"
    )

    # Step 2: construct all even-J/even-M basis states up to J_MAX.
    print("\nStep 1/5: building the |J,M> basis...")
    basis = build_basis(J_MAX)
    print(f"  basis size = {len(basis.states)} states")

    # Step 3: construct the complete angular interaction matrix.
    print("Step 2/5: building the full Q_x polarizability matrix...")
    qx = build_qx_matrix(basis)

    # Step 4: initialize the simulator and print ideal resonance locations.
    print("Step 3/5: preparing the split-operator propagator...")
    simulator = CentrifugeSimulator(basis, qx)

    print("  B-only |J,J> -> |J+2,J+2> resonance frequencies:")
    for J, resonance_ghz in ideal_raman_resonances(J_MAX):
        if F_FINAL_MIN_GHZ <= resonance_ghz <= F_FINAL_MAX_GHZ:
            print(f"    J={J:2d} -> {J+2:2d}: {resonance_ghz:9.4f} GHz")

    # Step 5: define the map and solve all TDSEs simultaneously.
    final_frequencies = np.linspace(
        F_FINAL_MIN_GHZ,
        F_FINAL_MAX_GHZ,
        N_FINAL_FREQUENCIES,
    )

    print(
        f"Step 4/5: propagating {len(final_frequencies)} independent "
        "centrifuges..."
    )
    results = simulator.simulate_many(final_frequencies)

    # Print a compact line for every calculation so failed adiabatic transfer or
    # basis truncation is visible without opening the figures.
    for number, result in enumerate(results, start=1):
        upper_boundary_population = result.final_p_j[-1]
        print(
            f"  [{number:3d}/{len(results)}] "
            f"f_final={result.final_frequency_ghz:7.2f} GHz  "
            f"<J>_final={result.final_mean_j:7.3f}  "
            f"J_dominant={result.final_dominant_j:2d}  "
            f"P(J_MAX)={upper_boundary_population:.2e}  "
            f"norm error={result.norm_error:.1e}"
        )

    # Step 6: generate plots and save data.
    print("Step 5/5: writing plots and numerical data...")
    prefix = output_directory / OUTPUT_PREFIX
    fan_path = prefix.with_name(prefix.name + "_fan.png")
    final_map_path = prefix.with_name(prefix.name + "_final_map.png")
    trajectory_path = prefix.with_name(prefix.name + "_trajectory.png")
    data_path = prefix.with_suffix(".npz")

    plot_fan(results, fan_path)
    plot_final_map(results, final_map_path)
    plot_single_trajectory(results[-1], trajectory_path)
    save_results(results, data_path)

    elapsed = wall_time.perf_counter() - start_time
    print(f"\nFinished in {elapsed:.1f} s")
    print(f"Saved {fan_path.name}")
    print(f"Saved {final_map_path.name}")
    print(f"Saved {trajectory_path.name}")
    print(f"Saved {data_path.name}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
