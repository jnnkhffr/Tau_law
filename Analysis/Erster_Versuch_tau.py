
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
    ["H2O"],
    ["CO2"],
    ["O3"],
    ["H2O", "CO2"],
    ["H2O", "CO2", "O3"]
]

KAYSER_GRID = np.linspace(1, 2000, 200)
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


def absorption_coefficient(atmosphere, species_list):
    absorbers = {sp: pa.recipe.SingleSpeciesAbsorption(species=sp) for sp in species_list}

    # defining atmospheric point
    atm_point = pa.arts.AtmPoint()
    temps = atmosphere["t"].values
    pressures = atmosphere["p"].values

    n_levels = len(temps)
    abs_dict = {sp: np.zeros((n_levels, len(FREQ_GRID))) for sp in species_list}

    # iterating through every level and calculating  absorption coefficient
    for h in range(n_levels):
        atm_point.temperature = temps[h]
        atm_point.pressure = pressures[h]

        for sp in species_list:
            atm_point[sp] = atmosphere[sp].values[h]
            abs_dict[sp][h] = absorbers[sp](FREQ_GRID, atm_point)

    # Summe aller Species
    total_abs = sum(abs_dict.values())

    return total_abs, abs_dict


def calculate_tau(abs_coeff):

    # thickness of layer, ordered from surface to TOA
    dz = np.diff(heights, prepend=heights[0])

    # Calculating tau from TOA towards ground
    tau = np.cumsum(abs_coeff[::-1, :] * dz[::-1, None], axis=0)

    # height where τ = 1 is reached, for every frequency
    tau_height = np.zeros(abs_coeff.shape[1]) # np.zeros(len(FREQ_GRID))

    for i in range(abs_coeff.shape[1]): # range(len(FREQ_GRID))

        # finding height index for where τ = 1 is reached
        idx = np.argmax(tau[:, i] >= 1) # index of height for τ = 1
        tau_height[i] = heights[-idx]

    return tau, tau_height


def set_up_workspace(atmosphere, species_list):

    ws = pa.Workspace()
    ws.absorption_speciesSet(
        species=species_list
    )
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

    pos = [100e3, 0, 0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()

    return ws


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

def calculate_total_flux(spectral_radiance):
    return np.trapezoid(spectral_radiance[:, 0], FREQ_GRID) * np.pi


def plot_tau_level(tau_height, filename, title_suffix=""):
    plt.figure()
    plt.plot(KAYSER_GRID, tau_height / 1000, "-")  # y axis in km
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Emission height at τ = 1 for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    # plt.savefig("C:/Users/janni/Desktop/Taulevel_CO2_H2O.pdf")
    # plt.savefig("Taulevel_CO2_H2O.pdf")
    plt.savefig(filename)
    # plt.savefig("Taulevel_CO2.pdf")

    plt.show()

def plot_tau_emission(tau_emission, filename, title_suffix=""):

    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, tau_emission)

    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"OLA at τ = 1 level for {title_suffix}")
    plt.grid(True, color='grey', linewidth=0.3)
    plt.savefig(filename)
    plt.show()


def main():

    # set up atmosphere
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    # set height levels to be used in claculation
    global heights
    heights = typhon.physics.pressure2height(pressure_levels)

    atmosphere = set_up_atmosphere(
        t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2
    )

    # Drei Fälle: nur H2O, nur CO2, beide zusammen
    SPECIES_CASES = [
        ["H2O"],
        ["CO2"],
        ["O3"],
        ["H2O", "CO2"],
        ["H20", "CO2", "O3"]
]

    for species_list in SPECIES_CASES:
        species_tag = "_".join(species_list)
        print("Berechne Fall:", species_list)


        # calculate absorption coefficient
        abs_total, abs_dict = absorption_coefficient(atmosphere, species_list)

        # calculate τ = 1 height and τ
        tau, tau_height = calculate_tau(abs_total)

        print("Tau shape:", tau.shape)
        print("Height where tau=1 (per frequency):", tau_height)

        # Radiance at tau=1 level
        tau_emission = spectral_radiance_at_tau_level(
            tau_height, atmosphere, species_list
        )
        print("tau emission: ", tau_emission)

        # Plots
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


if __name__ == "__main__":
    main()

