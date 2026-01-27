import time
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyarts3 as pa

import single_column_atmosphere as sca
import typhon

# Download ARTS catalogs if they are not already present
pa.data.download()

MIXING_RATIO_CO2 = 400e-6
MIXING_RATIO_O3 = 1e-6
T_SURF = 290  # K
SPECIES_CASES = [
    ["CO2"],
    ["H2O"],
    ["O3"],
    ["H2O", "CO2"],
    ["H2O", "CO2", "O3"]
]

KAYSER_GRID = np.linspace(1, 2000, 500)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)


def set_up_atmosphere(temp_profile, pressure_profile, H20_profile, CO2_concentration):
    atm = xr.Dataset(
        {
            "t": ("alt", temp_profile),
            "p": ("alt", pressure_profile),
            "H2O": ("alt", H20_profile),
            "O3": ('alt', np.ones_like(pressure_profile) * MIXING_RATIO_O3),
            "CO2": ('alt', np.ones_like(pressure_profile) * CO2_concentration),
        },
        coords={"alt": heights, "lat": 0, "lon": 0},
    )
    return atm


def calculate_all_species_absorption(atmosphere, all_species):
    """Calculate absorption for ALL species once, then combine as needed"""
    print("Calculating absorption coefficients for all species...")

    absorbers = {sp: pa.recipe.SingleSpeciesAbsorption(species=sp) for sp in all_species}

    atm_point = pa.arts.AtmPoint()
    temps = atmosphere["t"].values
    pressures = atmosphere["p"].values

    n_levels = len(temps)
    # Store absorption for each species separately
    abs_by_species = {sp: np.zeros((n_levels, len(FREQ_GRID))) for sp in all_species}

    for h in range(n_levels):
        atm_point.temperature = temps[h]
        atm_point.pressure = pressures[h]

        for sp in all_species:
            atm_point[sp] = atmosphere[sp].values[h]
            abs_by_species[sp][h] = absorbers[sp](FREQ_GRID, atm_point)

    print("Done calculating all species absorption coefficients")
    return abs_by_species


def combine_absorption(abs_by_species, species_list):
    """Combine pre-calculated absorption coefficients for requested species"""
    total_abs = np.zeros_like(abs_by_species[species_list[0]])
    abs_dict = {}

    for sp in species_list:
        abs_dict[sp] = abs_by_species[sp]
        total_abs += abs_by_species[sp]

    return total_abs, abs_dict


def calculate_tau(abs_coeff):
    """Optimized: vectorized operations"""
    dz = np.diff(heights, prepend=heights[0])

    # Vectorized tau calculation
    tau = np.cumsum(abs_coeff[::-1, :] * dz[::-1, None], axis=0)

    # Vectorized height finding
    tau_indices = np.argmax(tau >= 1, axis=0)
    tau_height = heights[-tau_indices]

    return tau, tau_height


def spectral_radiance_at_tau_level(tau_heights, atmosphere, species_list):
    ws = pa.Workspace()
    ws.absorption_speciesSet(
        species=species_list
    )
    ws.atmospheric_field = pa.data.to_atmospheric_field(atmosphere)

    ws.surface_fieldPlanet(option='Earth')
    ws.surface_field["t"] = atmosphere["t"].sel(alt=0).values

    ws.ReadCatalogData()

    cutoff = pa.arts.convert.kaycm2freq(25)
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = "ByLine"
        ws.absorption_bands[band].cutoff_value = cutoff

    ws.absorption_bands.keep_hitran_s(approximate_percentile=90)
    ws.propagation_matrix_agendaAuto()

    # Initialize output array: one radiance value per frequency
    tau_radiance = np.zeros(len(FREQ_GRID))

    # For each frequency, calculate radiance at its corresponding tau=1 height
    for i, height in enumerate(tau_heights):
        # Set observation position at the tau=1 height for this frequency
        pos = [height, 0, 0]
        los = [180.0, 0.0]  # looking upward

        # Set frequency grid to just this single frequency
        ws.frequency_grid = np.array([FREQ_GRID[i]])

        # Calculate ray path and radiance
        ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
        ws.spectral_radianceClearskyEmission()

        # Extract the radiance value (first frequency, first Stokes component)
        tau_radiance[i] = ws.spectral_radiance[0, 0]

    return tau_radiance


def spectral_radiance_at_toa(atmosphere, species_list):
    """Already efficient - single calculation"""
    ws = pa.Workspace()
    ws.absorption_speciesSet(species=species_list)
    ws.atmospheric_field = pa.data.to_atmospheric_field(atmosphere)

    ws.surface_fieldPlanet(option='Earth')
    ws.surface_field["t"] = atmosphere["t"].sel(alt=0).values

    ws.ReadCatalogData()

    cutoff = pa.arts.convert.kaycm2freq(25)
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = "ByLine"
        ws.absorption_bands[band].cutoff_value = cutoff

    ws.absorption_bands.keep_hitran_s(approximate_percentile=90)
    ws.propagation_matrix_agendaAuto()

    ws.frequency_grid = FREQ_GRID

    pos = [100e3, 0, 0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()
    return ws.spectral_radiance[:, 0]


def emission_by_temp(temperature):
    """Vectorized Planck function"""
    return typhon.physics.planck(FREQ_GRID, temperature)


def get_temperature_at_tau_heights(tau_heights, atmosphere):
    """Already optimized with numpy.interp"""
    atm_heights = atmosphere.coords['alt'].values
    atm_temps = atmosphere['t'].values
    return np.interp(tau_heights, atm_heights, atm_temps)


def calculate_total_flux(spectral_radiance):
    return np.trapezoid(spectral_radiance[:, 0], FREQ_GRID) * np.pi


def plot_tau_level(tau_height, filename, title_suffix=""):
    plt.figure()
    plt.plot(KAYSER_GRID, tau_height / 1000, "-", linewidth=0.7)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Emission height at τ = 1 for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}.pdf")
    plt.savefig(f"faster_plots/{filename}.svg", bbox_inches="tight")
    plt.close()


def plot_tau_emission(tau_emission, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, tau_emission, linewidth=0.7)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLR at τ = 1 level for {title_suffix}")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}.pdf")
    plt.savefig(f"faster_plots/{filename}.svg", bbox_inches="tight")
    plt.close()


def plot_planck_emission(planck_emission, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, planck_emission, linewidth=0.7)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"Planck emission at τ = 1 level for {title_suffix}")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}.pdf")
    plt.savefig(f"faster_plots/{filename}.svg", bbox_inches="tight")
    plt.close()


def plot_tau_level_scatter(tau_height, filename, title_suffix=""):
    plt.figure()
    plt.scatter(KAYSER_GRID, tau_height / 1000, s=1)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Emission height at τ = 1 for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}_scatter.pdf")
    plt.savefig(f"faster_plots/{filename}_scatter.svg", bbox_inches="tight")
    plt.close()


def plot_tau_emission_scatter(tau_emission, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.scatter(KAYSER_GRID, tau_emission, s=1)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLR at τ = 1 level for {title_suffix}")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}_scatter.pdf")
    plt.savefig(f"faster_plots/{filename}_scatter.svg", bbox_inches="tight")
    plt.close()


def plot_planck_emission_scatter(planck_emission, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.scatter(KAYSER_GRID, planck_emission, s=1)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"Planck emission at τ = 1 level for {title_suffix}")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}_scatter.pdf")
    plt.savefig(f"faster_plots/{filename}_scatter.svg", bbox_inches="tight")
    plt.close()


def plot_OLR_at_TOA(radiance, filename, title_suffix=""):
    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, radiance, linewidth=0.7)
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLR at TOA for {title_suffix}")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}.pdf")
    plt.savefig(f"faster_plots/{filename}.svg", bbox_inches="tight")
    plt.close()


def plot_OLR_diff(olr_diff, filename, title_suffix=""):
    fig, ax = plt.subplots()
    label = ["difference to ARTS", "difference to Planck"]
    for i in range(len(olr_diff)):
        ax.plot(KAYSER_GRID, olr_diff[i], linewidth=0.7, label=label[i])
    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLR at TOA minus OLR at emission height for \n {title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(True, color='grey', linewidth=0.3)
    plt.savefig(f"faster_plots/{filename}.pdf")
    plt.savefig(f"faster_plots/{filename}.svg", bbox_inches="tight")
    plt.close()


def main():
    start_time = time.time()

    # Set up atmosphere once
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    global heights
    heights = typhon.physics.pressure2height(pressure_levels)

    atmosphere = set_up_atmosphere(
        t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2
    )

    # OPTIMIZATION: Calculate absorption for ALL species ONCE
    all_species = ["H2O", "CO2", "O3"]
    abs_by_species = calculate_all_species_absorption(atmosphere, all_species)

    for species_list in SPECIES_CASES:
        species_tag = "_".join(species_list)
        case_start = time.time()
        print(f"\nBerechne Fall: {species_list}")

        # FAST: Just combine pre-calculated absorption coefficients
        abs_total, abs_dict = combine_absorption(abs_by_species, species_list)

        # Calculate τ = 1 height and τ
        tau, tau_height = calculate_tau(abs_total)
        print(f"Tau shape: {tau.shape}")

        # Emission temperature
        t_emission = get_temperature_at_tau_heights(tau_height, atmosphere)

        # Planck radiation with emission temperature
        planck_rad = emission_by_temp(t_emission)

        # Radiance at tau=1 level (ARTS) - OPTIMIZED
        tau_emission = spectral_radiance_at_tau_level(
            tau_height, atmosphere, species_list
        )

        # OLR at TOA
        olr_toa = spectral_radiance_at_toa(atmosphere, species_list)

        # OLR differences
        olr_diff_arts = olr_toa - tau_emission
        olr_diff_planck = olr_toa - planck_rad
        olr_diffs = [olr_diff_arts, olr_diff_planck]

        # Generate plots
        plot_tau_emission(tau_emission, f"Tau_Emission_{species_tag}", f"({species_tag})")
        plot_tau_level(tau_height, f"Taulevel_{species_tag}", f"({species_tag})")
        plot_planck_emission(planck_rad, f"Planck_Emission_{species_tag}", f"({species_tag})")

        # Scatter plots
        plot_tau_emission_scatter(tau_emission, f"Tau_Emission_{species_tag}", f"({species_tag})")
        plot_tau_level_scatter(tau_height, f"Taulevel_{species_tag}", f"({species_tag})")
        plot_planck_emission_scatter(planck_rad, f"Planck_Emission_{species_tag}", f"({species_tag})")
        plot_OLR_at_TOA(olr_toa, f"OLR_TOA_{species_tag}", f"({species_tag})")
        plot_OLR_diff(olr_diffs, f"OLR_diff_to_emission_level_for_{species_tag}", f"({species_tag})")

        case_time = time.time() - case_start
        print(f"Fall {species_tag} abgeschlossen in {case_time:.2f} Sekunden")

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nGesamtlaufzeit: {minutes} min {seconds:.1f} s")


if __name__ == "__main__":
    main()