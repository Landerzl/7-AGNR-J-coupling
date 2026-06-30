"""
Diagnosis of spin polarization in the AFM (singlet) and FM (triplet) states
of a finite 7-AGNR, to verify that the SCF cycle converges to the correct state.

Generates a panel with two molecule maps colored by m_i = n_up_i - n_dn_i.
- AFM state (singlet): zigzag edges should have opposite polarization (red vs blue).
- FM state (triplet): both edges should have the same polarization (same color).
"""

import sisl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from hubbard import sp2, HubbardHamiltonian, density

# ──────────────────── Parameters ────────────────────
L = 6          # Length in DBBA monomers (adjust as needed)
U_val = 3.0    # On-site repulsion (eV)
# ────────────────────────────────────────────────────

def build_7agnr_geometry(L):
    uc = sisl.geom.agnr(width=7, bond=1.42)
    geom = uc.tile(2 * L, axis=0)
    geom.set_nsc([1, 1, 1])
    while True:
        to_remove = []
        for ia in geom:
            idx = geom.close(ia, R=1.5)
            if len(idx) == 2:
                to_remove.append(ia)
        if not to_remove:
            break
        geom = geom.remove(to_remove)
    return geom

def get_bonds(geom):
    """Returns a list of pairs (i,j) with i<j for bonds with d < 1.5 Å."""
    bonds = []
    for ia in geom:
        idx = geom.close(ia, R=1.5)
        for j in idx:
            if j > ia:
                bonds.append((ia, j))
    return bonds

def plot_spin_map(ax, geom, m, bonds, title, vmax=None):
    """
    Plots the molecule with each atom colored by its local magnetic moment m_i.
    Red = spin up (m>0), Blue = spin down (m<0).
    """
    xyz = geom.xyz
    if vmax is None:
        vmax = max(np.max(np.abs(m)), 1e-6)

    # Draw bonds
    segments = []
    for (i, j) in bonds:
        segments.append([xyz[i, :2], xyz[j, :2]])
    lc = LineCollection(segments, colors='#888888', linewidths=1.2, zorder=1)
    ax.add_collection(lc)

    # Draw atoms colored by spin polarization
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=m, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    s=120, edgecolors='k', linewidths=0.5, zorder=2)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')

    # Annotate total spin
    Sz_total = np.sum(m) / 2.0
    ax.text(0.02, 0.95, f'Total $S_z$ = {Sz_total:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    return sc

# ── Build geometry ──
geom = build_7agnr_geometry(L)
bonds = get_bonds(geom)
xyz = geom.xyz
x_coords = xyz[:, 0]
tol_edge = 2.0
left_atoms  = np.where(x_coords < x_coords.min() + tol_edge)[0]
right_atoms = np.where(x_coords > x_coords.max() - tol_edge)[0]

print(f"7-AGNR with L={L} DBBA monomers, {geom.na} atoms")
print(f"Edges: {len(left_atoms)} left atoms, {len(right_atoms)} right atoms")

# ── 3NN TB Hamiltonian ──
H_tb = sp2(geom, t1=-2.7, t2=-0.2, t3=-0.18, spin='polarized')

# ══════════════════════════════════════════════════════
# AFM STATE (Open-shell Singlet)
# ══════════════════════════════════════════════════════
print("\n--- Converging AFM state (singlet) ---")
HH_afm = HubbardHamiltonian(H_tb, U=U_val, kT=1e-6, nkpt=[1, 1, 1])
HH_afm.set_polarization(left_atoms, dn=right_atoms)
dn_afm = HH_afm.converge(density.calc_n, tol=1e-13, print_info=True)
m_afm = HH_afm.n[0] - HH_afm.n[1]
E_afm = HH_afm.Etot
print(f"E_AFM = {E_afm:.10f} eV")
print(f"Total Sz (AFM) = {np.sum(m_afm)/2:.6f}")
print(f"|m| max (AFM) = {np.max(np.abs(m_afm)):.6f}")

# ══════════════════════════════════════════════════════
# FM STATE (Triplet)
# ══════════════════════════════════════════════════════
print("\n--- Converging FM state (triplet) ---")
N_tot = geom.na
q_fm = (N_tot // 2 + 1, N_tot // 2 - 1)
HH_fm = HubbardHamiltonian(H_tb, U=U_val, kT=1e-6, nkpt=[1, 1, 1], q=q_fm)
all_edge = np.concatenate((left_atoms, right_atoms))
HH_fm.set_polarization(all_edge)
dn_fm = HH_fm.converge(density.calc_n, tol=1e-13, print_info=True)
m_fm = HH_fm.n[0] - HH_fm.n[1]
E_fm = HH_fm.Etot
print(f"E_FM  = {E_fm:.10f} eV")
print(f"Total Sz (FM) = {np.sum(m_fm)/2:.6f}")
print(f"|m| max (FM) = {np.max(np.abs(m_fm)):.6f}")

J_meV = (E_fm - E_afm) * 1000
print(f"\n>>> J = E_FM - E_AFM = {J_meV:.6f} meV")

# ══════════════════════════════════════════════════════
# SPIN POLARIZATION PLOT
# ══════════════════════════════════════════════════════
# Use the same color scale for both panels
vmax = max(np.max(np.abs(m_afm)), np.max(np.abs(m_fm)))

fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)

sc1 = plot_spin_map(axes[0], geom, m_afm, bonds,
                    f'AFM State (Open-shell Singlet) — L={L}, E={E_afm:.6f} eV',
                    vmax=vmax)

sc2 = plot_spin_map(axes[1], geom, m_fm, bonds,
                    f'FM State (Triplet, $S_z$=1) — L={L}, E={E_fm:.6f} eV',
                    vmax=vmax)

# Shared colorbar
cbar = fig.colorbar(sc2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('$m_i = n_{\\uparrow} - n_{\\downarrow}$', fontsize=12)

fig.suptitle(f'Local Spin Polarization — 7-AGNR, L={L} DBBA monomers, U={U_val} eV',
             fontsize=15, fontweight='bold', y=1.02)

outfile = f'spin_polarization_L{L}.png'
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(f"\nPlot saved to '{outfile}'")
plt.show()
