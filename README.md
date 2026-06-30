# 7-AGNR Magnetic Coupling Analysis

This repository contains computational tools and scripts to study the magnetic exchange coupling ($J$) in finite 7-Armchair Graphene Nanoribbons (7-AGNRs). The project systematically compares two theoretical approaches to evaluate the magnetic interactions as a function of the nanoribbon length.

## Project Structure

*   **`Main/`**: Contains the core Python scripts for the simulations.
    *   `sisl_hubbard_7agnr.py`: The primary script of the project. It uses the `sisl` library to construct the finite 7-AGNR geometries and evaluates the exchange coupling $J$ using two methods:
        1.  **Effective Hubbard Dimer**: Extracts the effective hopping ($t_{\mathrm{eff}}$) and Coulomb repulsion ($U_{\mathrm{eff}}$) to estimate $J = 4t_{\mathrm{eff}}^2 / U_{\mathrm{eff}}$.
        2.  **Mean-Field Hubbard (MFH)**: Performs a self-consistent calculation to find the total energy difference between the Ferromagnetic (FM) and Antiferromagnetic (AFM) ground states ($J = E_{\mathrm{FM}} - E_{\mathrm{AFM}}$).
    *   `plot_spin_polarization_en.py`: Utility script to visualize the spin polarization localized at the zigzag edges of the nanoribbons.
    *   `test_geom.py`: A helper script to verify that the generated nanoribbon geometries are properly closed and have no unwanted dangling bonds.
    *   `Figures/`: Directory where the generated plots are saved (e.g., $J$ vs $L$ comparisons, difference between models, and spin distributions).
*   **`PDF/`**: Contains the LaTeX source (`main_results.tex`) and compiled PDF documents summarizing the main findings and figures of the project.
*   **`Ref/`**: Contains reference literature and background material, such as thesis documents related to the study.

## Key Results

The calculations yield comparative plots showing the exponential decay of the magnetic coupling $J$ with respect to the number of DBBA monomers ($L$). The plots include experimental reference scales, such as the superconducting gap of Nb(110) ($2\Delta \approx 3.0$ meV) and the thermal energy at standard STM operating temperatures ($1.2$ K), allowing for a direct assessment of whether the coupling falls within the Kondo regime or the superconducting gap.

## Dependencies

To run the scripts in this repository, ensure you have the following Python packages installed in your environment:

*   `numpy`
*   `scipy`
*   `matplotlib`
*   `sisl` (Required for building the geometries and setting up the advanced Tight-Binding Hamiltonians)

A local `hubbard` module is also required for the Mean-Field Hubbard calculations.
