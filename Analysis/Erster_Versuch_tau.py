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


def absorption_coefficient(atmosphere):

    h2o_absorption = pa.recipe.SingleSpeciesAbsorption(species="H2O")
    co2_absorption = pa.recipe.SingleSpeciesAbsorption(species="CO2")

    atm_point = pa.arts.AtmPoint()
    atm_point["CO2"] = MIXING_RATIO_CO2

    temps = atmosphere["t"].values
    pressures = atmosphere["p"].values
    h2o_values = atmosphere["H2O"].values

    n_levels = len(temps)

    absorption_h2o = np.zeros((n_levels, len(FREQ_GRID)))
    absorption_co2 = np.zeros((n_levels, len(FREQ_GRID)))

    for h in range(n_levels):
        atm_point.temperature = temps[h]
        atm_point.pressure = pressures[h]
        atm_point["H2O"] = h2o_values[h]

        absorption_h2o[h] = h2o_absorption(FREQ_GRID, atm_point)
        absorption_co2[h] = co2_absorption(FREQ_GRID, atm_point)

    return absorption_h2o, absorption_co2


def calculate_tau(abs_coeff):

    # thickness of layer, ordered from surface to TOA
    dz = np.diff(heights, prepend=heights[0])

    # Calculating tau from TOA towards ground
    tau = np.cumsum(abs_coeff[::-1, :] * dz[::-1, None], axis=0)

    # height when τ = 1 is reached, for every frequency
    tau_height = np.zeros(abs_coeff.shape[1]) # np.zeros(len(FREQ_GRID))

    for i in range(abs_coeff.shape[1]): # range(len(FREQ_GRID))

        idx = np.argmax(tau[:, i] >= 1) # index of height for τ = 1
        tau_height[i] = heights[-idx]

    return tau, tau_height


def set_up_workspace(atmosphere):

    ws = pa.Workspace()
    ws.absorption_speciesSet(
        species=["H2O", "CO2"]
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


def spectral_radiance_at_tau_level(tau_heights, atmosphere):

    fop = pa.recipe.SpectralAtmosphericFlux(
        species = ["H2O"],
        remove_lines_percentile = {"H2O": 70}
    )

    # atmosphere doesn't work
    flux, alts = fop(FREQ_GRID, atmosphere)

    # Vectorized height mapping
    height_idx = np.argmin(np.abs(heights[:, np.newaxis] - tau_heights), axis=0)

    # Vectorized indexing instead of loop
    freq_idx = np.arange(len(FREQ_GRID))

    # radiance for each height and frequency
    tau_radiance = flux.up.T[height_idx, freq_idx]

    return tau_radiance

def calculate_total_flux(spectral_radiance):
    return np.trapezoid(spectral_radiance[:, 0], FREQ_GRID) * np.pi


def plot_tau_level(tau_height):
    plt.figure()
    plt.plot(KAYSER_GRID, tau_height / 1000, "-")  # y axis in km
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title("Emission height (τ = 1) vs. wavenumber for CO$_2$")
    plt.grid(True, color='grey', linewidth=0.3)
    # plt.savefig("C:/Users/janni/Desktop/Taulevel_CO2_H2O.pdf")
    # plt.savefig("Taulevel_CO2_H2O.pdf")
    plt.savefig("Taulevel_H2O.pdf")
    # plt.savefig("Taulevel_CO2.pdf")

    plt.show()


def main():

    # set up atmosphere
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    global heights
    heights = typhon.physics.pressure2height(pressure_levels)

    atmosphere = set_up_atmosphere(t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2)

    # calculate absorption coefficient
    h2o, co2 = absorption_coefficient(atmosphere)
    #abs_coeff = h2o + co2
    #abs_coeff = co2
    abs_coeff = h2o

    # calculate τ = 1 height and τ
    tau, tau_height = calculate_tau(abs_coeff)

    print("Tau shape:", tau.shape)
    print("Height where tau=1 (per frequency):", tau_height)
    tau_emission = spectral_radiance_at_tau_level(tau_height, atmosphere)
    print("tau emission: ", tau_emission)
    # Plot tau = 1 height vs frequency
    #plot_tau_level(tau_height)


if __name__ == "__main__":
    main()

