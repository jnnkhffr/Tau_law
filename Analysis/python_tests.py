import numpy as np
import typhon
import xarray as xr
import single_column_atmosphere as sca
NUMBER_OF_PRESSURE_LEVELS = 100
PRESSURE_LEVELS = np.logspace(3, 0, NUMBER_OF_PRESSURE_LEVELS)
heights = typhon.physics.pressure2height(PRESSURE_LEVELS)
T_SURF = 290  # K
MIXING_RATIO_CO2 = 400e-6
MIXING_RATIO_O3 = 1e-6
T_SURF = 290  # K

tau_heights = np.array([1000, 2000, 3000, 22000,40000])
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

t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
# set height levels to be used in claculation


atmosphere = set_up_atmosphere(
    t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2
)

def get_temperature_at_tau_heights(tau_heights, atmosphere):
    """
    Interpolate atmospheric temperature at tau=1 heights using numpy.

    Parameters:
    -----------
    tau_heights : array-like
        Heights where tau=1 is reached for each frequency (in meters)
    atmosphere : xr.Dataset
        Atmospheric dataset containing 't' (temperature) and 'alt' (altitude) coords

    Returns:
    --------
    tau_temperatures : np.ndarray
        Temperatures at each tau=1 height (in Kelvin)
    """
    # Get atmosphere height and temperature profiles
    atm_heights = atmosphere.coords['alt'].values
    atm_temps = atmosphere['t'].values

    # Use numpy's interp function for linear interpolation
    # np.interp automatically handles extrapolation by using edge values
    tau_temperatures = np.interp(tau_heights, atm_heights, atm_temps)

    return tau_temperatures

print(get_temperature_at_tau_heights(tau_heights, atmosphere))