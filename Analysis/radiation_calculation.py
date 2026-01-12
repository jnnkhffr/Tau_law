import os

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

T_SURF = 290 # K

KAYSER_GRID = np.linspace(1, 2000, 200)  # in Kayser (cm^-1)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)

def set_up_atmosphere(temp_profile, pressure_profile, H20_profile, CO2_concentration):
    '''
    Returns a xarray dataset of our atmosphere for the ARTS calculation.

    :param temp_profile: temperature profile
    :param pressure_profile: pressure profile
    :param H20_profile: water vapor mixing ratio profile
    :param CO2_concentration: CO2 concentration
    :return: xarray dataset of our atmosphere profile
    '''

    #
    heights = typhon.physics.pressure2height(pressure_profile)

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

    atm["t"].attrs = {
        "units": "K",
        "long_name": "Temperature",
    }
    atm["p"].attrs = {
        "units": "Pa",
        "long_name": "Pressure",
    }
    atm["H2O"].attrs = {
        "units": "mol/mol",
        "long_name": "Water vapor volume mixing ratio",
    }
    atm["O3"].attrs = {
        "units": "mol/mol",
        "long_name": "Ozone volume mixing ratio",
    }
    atm["CO2"].attrs = {
        "units": "mol/mol",
        "long_name": "CO2 mixing ratio",
    }
    atm["alt"].attrs = {
        "units": "m",
        "long_name": "Geometric altitude",
    }
    return atm

def set_up_workspace(atmosphere):

    # Create a pyarts workspace
    ws = pa.Workspace()
    ws.absorption_speciesSet(
        species=["H2O", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2", "O3"]
    )
    ws.atmospheric_field = pa.data.to_atmospheric_field(atmosphere)

    # remember to update the surface temperature too
    ws.surface_fieldPlanet(option='Earth')
    ws.surface_field["t"] = atmosphere["t"].sel(alt=0).values

    # Set up frequency grid

    ws.frequency_grid = pa.arts.convert.kaycm2freq(KAYSER_GRID)  # in Hz

    # Select absorption species and continuum model
    # This example uses a reduced set of species to speed up the calculation.
    # Use the second line for a more realistic setup.
    # ws.absorption_speciesSet(
    #   species=["H2O-161, H2O-ForeignContCKDMT400, H2O-SelfContCKDMT400", "CO2-626"]
    # )
    # ws.absorption_speciesSet(
    #     species=["H2O, H2O-ForeignContCKDMT400, H2O-SelfContCKDMT400", "CO2", "O3"]
    # )

    # Read spectral line data from ARTS catalog
    ws.ReadCatalogData()

    # Apply a frequency cutoff. To be consistent with the CKD water vapor continuum,
    # a cutoff of 25 Kayser is necessary. We set it here for all species, because it
    # also speeds up the calculation.
    cutoff = pa.arts.convert.kaycm2freq(25)
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = "ByLine"
        ws.absorption_bands[band].cutoff_value = cutoff

    # Remove 90% of the lines to speed up the calculation
    ws.absorption_bands.keep_hitran_s(approximate_percentile=90)

    # Automatically set up the methods to compute absorption coefficients
    ws.propagation_matrix_agendaAuto()

    # this doesn't raise an error but I have no idea what it does
    pa.recipe.SingleSpeciesAbsorption(species="H2O")

    # Set up geometry of observation
    pos = [100e3, 0, 0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()
    return ws

def calculate_spectral_radiance(workspace):
    return workspace.spectral_radiance

def calculate_total_flux(spectral_radiance):
    return np.trapezoid(spectral_radiance[:, 0], FREQ_GRID) * np.pi

def calculate_absorption_coefficient(pressure_profile, workspace):

    # this doesn't raise an error, but I have no idea what it does
    absorption = pa.recipe.SingleSpeciesAbsorption(species="H2O")

    heights = typhon.physics.pressure2height(pressure_profile)
    absorption_coefficient = []

    for h in heights:
        atm_point = workspace.atmospheric_field(h, 0, 0)
        print(f"atm_point: {atm_point}")
        absorption_coefficient.append(absorption(FREQ_GRID, atm_point))
        print(f"absorption coefficient: {absorption_coefficient}")

    absorption_coefficient = np.array(absorption_coefficient)
    return absorption_coefficient

def plot_ola(spectral_radiance, flux):
    fig, ax = plt.subplots()
    ax.plot(KAYSER_GRID, spectral_radiance[:, 0])

    # for t in temperatures:
    #    ax.plot(kayser_grid, typhon.physics.planck(freq_grid, t), label=f"T={t} K")

    ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance ($Wm^{-2}sr^{-1}Hz^{-1}$)")
    ax.set_title(f"Clear sky OLA radiance, O3_mr = {MIXING_RATIO_O3}, CO2_mr = {MIXING_RATIO_CO2}")
    ax.text(
        0.98, 0.98,  # X and Y coordinates (in axes fraction)
        f"Total OLA = {flux:.2f} W/m²",  # Text to display
        transform=ax.transAxes,  # Use axes coordinates (0–1)
        fontsize=10,  # Font size
        verticalalignment='top',  # Align top
        horizontalalignment='right')

    # ax.legend()
    # plt.savefig(f"Ex06/OLA_CO2_{MIXING_RATIO_CO2}.pdf")

    if "ARTS_HEADLESS" not in os.environ:
        plt.show()


def main():

    # set up atmosphere
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    atmosphere = set_up_atmosphere(t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2)

    # radiation calculations
    working_space = set_up_workspace(atmosphere) #pyarts workspace with our atmosphere
    spectral_radiance_toa = calculate_spectral_radiance(working_space)

    total_flux = calculate_total_flux(spectral_radiance_toa)
    absorption_coeff = calculate_absorption_coefficient(pressure_levels, working_space)

    # plot results
    #plot_ola(spectral_radiance_toa, total_flux)

if __name__ == "__main__":
    main()