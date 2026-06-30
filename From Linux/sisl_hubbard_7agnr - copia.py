"""
Magnetic coupling J in finite 7-AGNRs (L = 4..20 DBBA monomers)
via two methods:
  1) Mean-Field Hubbard (MFH): J = E_FM - E_AFM
  2) Effective Hubbard dimer:  J = 4 t_eff^2 / U_eff

Generates three plots:
  - J_MFH_vs_L.png          (Mean-Field Hubbard only)
  - J_dimer_vs_L.png         (Hubbard dimer only)
  - J_comparison_vs_L.png    (both methods compared)
"""

import sisl
import numpy as np
import matplotlib.pyplot as plt
from hubbard import sp2, HubbardHamiltonian, density

# ═══════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════

def build_7agnr_geometry(L):
    """
    Build a finite 7-AGNR geometry with L DBBA monomers.
    1 monomer = 2 unit cells.  Iterative pruning of dangling bonds.
    """
    uc = sisl.geom.agnr(width=7, bond=1.42)
    geom = uc.tile(2 * L, axis=0)
    geom.set_nsc([1, 1, 1])  # Remove periodicity -> finite molecule

    while True:
        to_remove = []
        for ia in geom:
            idx = geom.close(ia, R=1.5)
            if len(idx) == 2:  # self + 1 neighbour = coordination 1
                to_remove.append(ia)
        if not to_remove:
            break
        geom = geom.remove(to_remove)
    return geom

# ═══════════════════════════════════════════════════════════
# METHOD 1 — EFFECTIVE HUBBARD DIMER
# ═══════════════════════════════════════════════════════════

def calculate_J_dimer(geom, U_bare=3.0, t1=-2.7, t2=-0.2, t3=-0.18):
    """
    Project the 7-AGNR physics onto an effective Hubbard dimer.

    t_eff  = Delta_gap / 2   (hybridisation gap of the pure TB Hamiltonian)
    U_eff  = U_bare * sum_i |psi_L(i)|^4
             where psi_L = (psi_HOMO + psi_LUMO) / sqrt(2) is the
             localised end state (SPTE)
    J      = 4 t_eff^2 / U_eff
    """
    # 3NN TB Hamiltonian without spin polarisation
    H_tb = sp2(geom, t1=t1, t2=t2, t3=t3, spin='unpolarized')

    # Full diagonalisation
    evals, evecs = H_tb.eigh(eigvals_only=False)
    N = len(evals)
    idx_homo = N // 2 - 1
    idx_lumo = N // 2

    # t_eff = half the hybridisation gap
    t_eff = np.abs(evals[idx_lumo] - evals[idx_homo]) / 2.0

    psi_homo = evecs[:, idx_homo]
    psi_lumo = evecs[:, idx_lumo]

    # Reconstruct the localised end state
    psi_L = (psi_homo + psi_lumo) / np.sqrt(2)
    U_eff = U_bare * np.sum(np.abs(psi_L)**4)

    J_dimer = 4.0 * t_eff**2 / U_eff

    return {
        'J': J_dimer,
        't_eff': t_eff,
        'U_eff': U_eff,
    }

# ═══════════════════════════════════════════════════════════
# METHOD 2 — MEAN-FIELD HUBBARD  (E_FM - E_AFM)
# ═══════════════════════════════════════════════════════════

def calculate_J_mfh(geom, U_val=3.0, t1=-2.7, t2=-0.2, t3=-0.18):
    """
    Compute J = E_FM - E_AFM from the self-consistent MFH cycle.
    """
    H_tb = sp2(geom, t1=t1, t2=t2, t3=t3, spin='polarized')

    # Identify zigzag-edge atoms at the two ends
    xyz = geom.xyz
    x_coords = xyz[:, 0]
    tol_edge = 2.0
    left_atoms  = np.where(x_coords < x_coords.min() + tol_edge)[0]
    right_atoms = np.where(x_coords > x_coords.max() - tol_edge)[0]

    # -- AFM state (open-shell singlet) --
    HH_afm = HubbardHamiltonian(H_tb, U=U_val, kT=1e-6, nkpt=[1, 1, 1])
    HH_afm.set_polarization(left_atoms, dn=right_atoms)
    HH_afm.converge(density.calc_n, tol=1e-14, print_info=False)
    E_afm = HH_afm.Etot

    # -- FM state (triplet, Sz = 1) --
    N_tot = geom.na
    q_fm = (N_tot // 2 + 1, N_tot // 2 - 1)
    HH_fm = HubbardHamiltonian(H_tb, U=U_val, kT=1e-6, nkpt=[1, 1, 1], q=q_fm)
    all_edge = np.concatenate((left_atoms, right_atoms))
    HH_fm.set_polarization(all_edge)
    HH_fm.converge(density.calc_n, tol=1e-14, print_info=False)
    E_fm = HH_fm.Etot

    return {
        'J': E_fm - E_afm,
        'E_afm': E_afm,
        'E_fm': E_fm,
    }

# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    L_vals = list(range(4, 11))
    U_bare = 3.0

    results_dimer = []
    results_mfh   = []

    header = (f"{'L':<4} | {'N_at':<5} | "
              f"{'t_eff(meV)':<11} | {'U_eff(eV)':<10} | {'J_dim(meV)':<11} | "
              f"{'E_AFM(eV)':<14} | {'E_FM(eV)':<14} | {'J_MFH(meV)':<11}")
    print(header)
    print("─" * len(header))

    for L in L_vals:
        geom = build_7agnr_geometry(L)

        # -- Hubbard dimer --
        rd = calculate_J_dimer(geom, U_bare=U_bare)
        results_dimer.append(rd)

        # -- Mean-Field Hubbard --
        try:
            rm = calculate_J_mfh(geom, U_val=U_bare)
        except Exception as e:
            rm = {'J': np.nan, 'E_afm': np.nan, 'E_fm': np.nan}
            print(f"  [MFH L={L} ERROR: {e}]")
        results_mfh.append(rm)

        print(f"{L:<4} | {geom.na:<5} | "
              f"{rd['t_eff']*1000:<11.4f} | {rd['U_eff']:<10.4f} | {rd['J']*1000:<11.4e} | "
              f"{rm['E_afm']:<14.6f} | {rm['E_fm']:<14.6f} | {rm['J']*1000:<11.4e}")

    # Convert to arrays for plotting
    L_arr = np.array(L_vals)
    J_dimer_meV = np.array([r['J'] for r in results_dimer]) * 1000
    J_mfh_meV   = np.array([r['J'] for r in results_mfh])   * 1000

    # ── Exponential fit to dimer data: J = A * exp(-L / xi) ──
    # Use log-linear fit:  ln(J) = ln(A) - L/xi
    log_J = np.log(J_dimer_meV)
    coeffs = np.polyfit(L_arr, log_J, 1)   # slope = -1/xi, intercept = ln(A)
    xi = -1.0 / coeffs[0]                  # decay length in DBBA monomers
    A_fit = np.exp(coeffs[1])
    L_fit = np.linspace(L_arr[0], L_arr[-1], 200)
    J_fit = A_fit * np.exp(-L_fit / xi)

    print(f"\n── Exponential fit: J = {A_fit:.3e} meV * exp(-L / {xi:.3f})")
    print(f"   Decay length xi = {xi:.3f} DBBA monomers")

    # ── MFH: separate positive (physical) and negative (noise) points ──
    J_mfh_abs = np.abs(J_mfh_meV)
    mfh_finite = np.isfinite(J_mfh_abs) & (J_mfh_abs > 0)
    mfh_pos = mfh_finite & (J_mfh_meV > 0)
    mfh_neg = mfh_finite & (J_mfh_meV < 0)

    # ═════════════════════════════════════════════════════
    #  PLOT 1 — Mean-Field Hubbard only
    # ═════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(L_arr[mfh_pos], J_mfh_abs[mfh_pos],
             'o-', lw=2.5, color='#d62728', markersize=8,
             markerfacecolor='white', markeredgewidth=2,
             label='MFH: $J = E_{\\mathrm{FM}} - E_{\\mathrm{AFM}}$')
    if np.any(mfh_neg):
        ax1.plot(L_arr[mfh_neg], J_mfh_abs[mfh_neg],
                 'x', color='#d62728', markersize=10, markeredgewidth=2.5,
                 label='MFH: $|J|$ (negative sign — numerical noise)')
    ax1.set_yscale('log')
    ax1.set_xlabel('Length $L$ (DBBA monomers)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$|J|$ (meV)', fontsize=12, fontweight='bold')
    ax1.set_title('Magnetic Coupling — Mean-Field Hubbard', fontsize=14, pad=12)
    ax1.set_xticks(L_vals)
    ax1.grid(True, which='major', ls='-', alpha=0.4)
    ax1.grid(True, which='minor', ls='--', alpha=0.15)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    fig1.savefig('J_MFH_vs_L.png', dpi=300, bbox_inches='tight')
    print("\n-> Saved 'J_MFH_vs_L.png'")

    # ═════════════════════════════════════════════════════
    #  PLOT 2 — Hubbard dimer only (with fit)
    # ═════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(L_vals, J_dimer_meV,
             's', color='#1f77b4', markersize=8,
             markerfacecolor='white', markeredgewidth=2, zorder=3,
             label='Dimer: $J = 4t_{\\mathrm{eff}}^2 \\,/\\, U_{\\mathrm{eff}}$')
    ax2.plot(L_fit, J_fit, '--', lw=2, color='#1f77b4', alpha=0.7,
             label=f'Fit: $J = A\\, e^{{-L/\\xi}}$,  $\\xi = {xi:.2f}$ DBBA monomers')
    ax2.set_yscale('log')
    ax2.set_xlabel('Length $L$ (DBBA monomers)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$J$ (meV)', fontsize=12, fontweight='bold')
    ax2.set_title('Magnetic Coupling — Effective Hubbard Dimer', fontsize=14, pad=12)
    ax2.set_xticks(L_vals)
    ax2.grid(True, which='major', ls='-', alpha=0.4)
    ax2.grid(True, which='minor', ls='--', alpha=0.15)
    ax2.legend(fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    fig2.savefig('J_dimer_vs_L.png', dpi=300, bbox_inches='tight')
    print("-> Saved 'J_dimer_vs_L.png'")

    # ═════════════════════════════════════════════════════
    #  PLOT 3 — Comparison (with fit and noise floor)
    # ═════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Dimer data + fit
    ax3.plot(L_vals, J_dimer_meV,
             's', color='#1f77b4', markersize=8,
             markerfacecolor='white', markeredgewidth=2, zorder=3,
             label='Dimer: $J = 4t_{\\mathrm{eff}}^2 / U_{\\mathrm{eff}}$')
    ax3.plot(L_fit, J_fit, '--', lw=2, color='#1f77b4', alpha=0.6,
             label=f'Fit: $\\xi = {xi:.2f}$ DBBA monomers')

    # MFH positive points (connected)
    ax3.plot(L_arr[mfh_pos], J_mfh_abs[mfh_pos],
             'o-', lw=2, color='#d62728', markersize=8,
             markerfacecolor='white', markeredgewidth=2,
             label='MFH: $J = E_{\\mathrm{FM}} - E_{\\mathrm{AFM}}$')

    # MFH negative points (noise)
    if np.any(mfh_neg):
        ax3.plot(L_arr[mfh_neg], J_mfh_abs[mfh_neg],
                 'x', color='#d62728', markersize=10, markeredgewidth=2.5,
                 label='MFH: $|J|$ (sign flip — noise)')

    # Mark the noise floor
    if np.any(mfh_neg):
        noise_floor = np.median(J_mfh_abs[~mfh_pos & mfh_finite])
        ax3.axhline(noise_floor, ls=':', color='gray', alpha=0.6, lw=1.5)
        ax3.text(L_arr[-1] + 0.3, noise_floor, 'SCF noise floor',
                 fontsize=9, color='gray', va='center')

    # Add superconducting gap reference
    two_delta = 3.0 # meV, gap for Nb(110)

    ax3.axhline(two_delta, color='purple', ls='-.', lw=1.5, label='Gap SC $2\\Delta$ Nb(110) (3.0 meV)')

    ax3.set_yscale('log')

    # Determine bounds based on actual data to avoid log-scale cutoff issues
    data_max = max(np.max(J_dimer_meV), np.max(J_mfh_abs[mfh_pos]) if np.any(mfh_pos) else 0)
    data_min = min(np.min(J_dimer_meV), np.min(J_mfh_abs[mfh_pos]) if np.any(mfh_pos) else 1)
    
    axh_top = max(data_max * 2, two_delta * 5)
    axh_bottom = min(data_min * 0.5, two_delta * 0.5)
    
    # Apply robust limits
    ax3.set_ylim(bottom=axh_bottom, top=axh_top)
    ax3.set_xlabel('Length $L$ (DBBA monomers)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('$|J|$ (meV)', fontsize=12, fontweight='bold')
    ax3.set_title('Magnetic Coupling Comparison — 7-AGNR', fontsize=14, pad=12)
    ax3.set_xticks(L_vals)
    ax3.grid(True, which='major', ls='-', alpha=0.4)
    ax3.grid(True, which='minor', ls='--', alpha=0.15)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.tight_layout()
    fig3.savefig('J_comparison_vs_L.png', dpi=300, bbox_inches='tight')
    print("-> Saved 'J_comparison_vs_L.png'")

    plt.show()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()

