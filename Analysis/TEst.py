"""
Calculate the clear sky upwelling longwave radiance at the top of the atmosphere. The atmosphere
is a standard tropical atmosphere, containing only water vapor, carbon dioxide and ozone
as trace gases. The water vapor absorption continuum is included.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import typhon
import pyarts3 as pa

import single_column_atmosphere as sca
pa.data.download()

MIXING_RATIO_CO2 = 400e-6
MIXING_RATIO_O3 = 1e-6
T_SURF = 290  # K

KAYSER_GRID = np.linspace(1, 2000, 200)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)

def set_up_atmosphere(temp_profile, pressure_profile, H20_profile, CO2_concentration):

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

    return atm

t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
atmosphere = set_up_atmosphere(t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2)


heights = typhon.physics.pressure2height(pressure_levels)

fop = pa.recipe.SpectralAtmosphericFlux(
    species=["H2O"],
    remove_lines_percentile={"H2O": 70}
)

# First, let's see what the original atmosphere looks like
original_atm = fop.get_atmosphere()
print("Original atmosphere keys:", original_atm.keys())
print("Original atmosphere length:", len(original_atm['t']))

# Now create your atmosphere with ALL the required fields matching that length
n_levels = len(atmosphere['t'].values)

atm = {
    't': atmosphere['t'].values.tolist(),
    'H2O': atmosphere['H2O'].values.tolist(),
    'p': atmosphere['p'].values.tolist(),
    'wind_u': [0] * n_levels,
    'wind_v': [0] * n_levels,
    'wind_w': [0] * n_levels,
    'mag_u': [0] * n_levels,
    'mag_v': [0] * n_levels,
    'mag_w': [0] * n_levels,
}

# Add any other fields that were in original_atm
for key in original_atm.keys():
    if key not in atm:
        atm[key] = [0] * n_levels

print("New atm keys:", atm.keys())
print("New atm length:", len(atm['t']))

flux, alt = fop(FREQ_GRID, atm)