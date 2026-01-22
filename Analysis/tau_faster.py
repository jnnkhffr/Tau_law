import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyarts3 as pa
from numpy import dtype, float64, ndarray

import single_column_atmosphere as sca
import typhon

# Download ARTS catalogs if they are not already present
pa.data.download()

MIXING_RATIO_CO2 = 400e-6
MIXING_RATIO_O3 = 1e-6
T_SURF = 290  # K
SPECIES_CASES = [
    ["H2O"],
    ["CO2"],
    ["O3"],
    ["H2O", "CO2"],
    ["H2O", "CO2", "O3"]  # Fixed typo: "H20" -> "H2O"
]

KAYSER_GRID = np.linspace(1, 2000, 200)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)


def set_up_atmosphere(temp_profile, pressure_profile, H2O_profile, CO2_concentration):
    atm = xr.Dataset(
        {
            "t": ("alt", temp_profile),
            "p": ("alt", pressure_profile),
            "H2O": ("alt", H2O_profile),
            "O3": ('alt', np.ones_like(pressure_profile) * MIXING_RATIO_O3),
            "CO2": ('alt', np.ones_like(pressure_profile) * CO2_concentration),
        },
        coords={"alt": heights, "lat": 0, "lon": 0},
    )
    return atm


def absorption_coefficient_all_species(atmosphere, all_species):
    """Calculate absorption coefficients for all species at once"""
    absorbers = {sp: pa.recipe.SingleSpeciesAbsorption(species=sp) for sp in all_species}

    atm_point = pa.arts.AtmPoint()
    temps = atmosphere["t"].values
    pressures = atmosphere["p"].values

    n_levels = len(temps)
    abs_dict = {sp: np.zeros((n_levels, len(FREQ_GRID))) for sp in all_species}

    # Calculate absorption for all species at once
    for h in range(n_levels):
        atm_point.temperature = temps[h]
        atm_point.pressure = pressures[h]

        for sp in all_species:
            atm_point[sp] = atmosphere[sp].values[h]
            abs_dict[sp][h] = absorbers[sp](FREQ_GRID, atm_point)

    return abs_dict


def calculate_tau(abs_coeff):
    """Calculate optical depth and tau=1 height"""
    dz = np.diff(heights, prepend=heights[0])
    tau = np.cumsum(abs_coeff[::-1, :] * dz[::-1, None], axis=0)

    tau_height = np.zeros(abs_coeff.shape[1])
    for i in range(abs_coeff.shape[1]):
        idx = np.argmax(tau[:, i] >= 1)
        tau_height[i] = heights[-idx] if idx > 0 else heights[-1]

    return tau, tau_height


def set_up_workspace_once(atmosphere, all_species):
    """Set up workspace once with all species"""
    ws = pa.Workspace()
    ws.absorption_speciesSet(species=all_species)
    ws.atmospheric_field = pa.data.to_atmospheric_field(atmosphere)

    ws.surface_fieldPlanet(option='Earth')
    ws.surface_field["t"] = atmosphere["t"].sel(alt=0).values

    ws.frequency_grid = FREQ_GRID

    ws.ReadCatalogData()

    cutoff = pa.arts.convert.kaycm2freq(25)
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = "ByLine"
        ws.absorption_bands[band].cutoff_value = cutoff

    ws.absorption_bands.keep_hitran_s(approximate_percentile=90)
    ws.propagation_matrix_agendaAuto()

    return ws


def spectral_radiance_at_all_heights(unique_heights, atmosphere, all_species, ws):
    """Calculate spectral radiance at all unique heights at once"""
    radiance_cache = {}

    for height in unique_heights:
        pos = [height, 0, 0]
        los = [180.0, 0.0]

        ws.frequency_grid = FREQ_GRID
        ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
        ws.spectral_radianceClearskyEmission()

        # Store full spectral radiance for this height
        radiance_cache[height] = ws.spectral_radiance[:, 0].copy()

    return radiance_cache


def plot_tau_level(tau_height, filename, title_suffix=""):
    plt.figure()
    plt.plot(KAYSER_GRID, tau_height / 1000, "-")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Emission height (τ = 1) vs. wavenumber for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    plt.savefig(filename)
    plt.close()


def plot_tau_emission(tau_emission, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, tau_emission)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLA at τ = 1 level for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    plt.savefig(filename)
    plt.close()


def main():
    # Set up atmosphere
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)

    global heights
    heights = typhon.physics.pressure2height(pressure_levels)

    atmosphere = set_up_atmosphere(
        t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2
    )

    # Get all unique species across all cases
    all_species = list(set(sp for case in SPECIES_CASES for sp in case))
    print("Computing absorption coefficients for all species:", all_species)

    # COMPUTE ABSORPTION COEFFICIENTS ONLY ONCE
    abs_dict_all = absorption_coefficient_all_species(atmosphere, all_species)

    # SET UP WORKSPACE ONLY ONCE
    ws = set_up_workspace_once(atmosphere, all_species)

    # Collect all unique tau heights needed
    tau_heights_per_case = {}

    for species_list in SPECIES_CASES:
        # Combine absorption coefficients for this case
        abs_total = sum(abs_dict_all[sp] for sp in species_list)
        tau, tau_height = calculate_tau(abs_total)
        tau_heights_per_case[tuple(species_list)] = tau_height

    # Get all unique heights
    all_tau_heights = set()
    for tau_height in tau_heights_per_case.values():
        all_tau_heights.update(tau_height)

    unique_heights = sorted(all_tau_heights)
    print(f"Computing spectral radiance at {len(unique_heights)} unique heights...")

    # COMPUTE SPECTRAL RADIANCE ONLY ONCE FOR ALL UNIQUE HEIGHTS
    radiance_cache = spectral_radiance_at_all_heights(unique_heights, atmosphere, all_species, ws)

    # Now process each case using cached values
    for species_list in SPECIES_CASES:
        species_tag = "_".join(species_list)
        print(f"\nProcessing case: {species_list}")

        tau_height = tau_heights_per_case[tuple(species_list)]

        # Extract tau emission from cache
        tau_emission = np.array([radiance_cache[h][i] for i, h in enumerate(tau_height)])

        print(f"Tau emission range: [{tau_emission.min():.2e}, {tau_emission.max():.2e}]")

        # Generate plots
        plot_tau_emission(
            tau_emission,
            filename=f"Tau_Emission_{species_tag}.pdf",
            title_suffix=f"({species_tag})",
        )
        plot_tau_level(
            tau_height,
            filename=f"Taulevel_{species_tag}.pdf",
            title_suffix=f"({species_tag})",
        )

    print("\nAll cases processed successfully!")


if __name__ == "__main__":
    main()