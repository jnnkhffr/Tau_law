import matplotlib.pyplot as plt
import numpy as np
import typhon

#SURFACE_TEMPERATURE = 290                 # K
t_0 = 290                                  # K
COLD_POINT_TEMPERATURE_TROPOPAUSE = 200    # K
RELATIVE_HUMIDITY = 0.8                    # between 0 and 1
NUMBER_OF_PRESSURE_LEVELS = 100
PRESSURE_LEVELS = np.logspace(3, 0, NUMBER_OF_PRESSURE_LEVELS)  * 10 ** 2 # in Pa

def create_vertical_profile(surface_temperature: float):
    '''
    Sets up basic atmosphere profiles for temperature and water vapor mixing ratio for every pressure level.
    Takes surface temperature as argument.
    :param surface_temperature: Surface temperature in Kelvin
    :return: temperature profile  and water_vapor_volume_mixing_ratio
    '''

    # create empty arrays to store data
    water_vapor_volume_mixing_ratio = np.zeros_like(PRESSURE_LEVELS)
    t_profile = np.zeros_like(PRESSURE_LEVELS)

    # pressure levels into height levels in m
    heights = typhon.physics.pressure2height(PRESSURE_LEVELS)

    # set first value to surface temperature
    t_profile[0] = surface_temperature  # t_0 is set to surface pressure to start from
    water_vapor_volume_mixing_ratio[0] = typhon.physics.relative_humidity2vmr(RELATIVE_HUMIDITY, PRESSURE_LEVELS[0], t_profile[0])

    first_tp_p_lvl = True # first pressure level of tropopause

    # iterate through pressure levels
    for i in range(1, len(PRESSURE_LEVELS)):

        # Temperature gradient
        lapse_rate = typhon.physics.moist_lapse_rate(PRESSURE_LEVELS[i-1], t_profile[i-1]) # (Kelvin/Meter)

        # Thickness of height levels
        delta_z = heights[i] - heights[i-1]

        # new temperature using lapse rate
        t_profile[i] = t_profile[i-1] - lapse_rate * delta_z

        # new wmr using new temperature
        water_vapor_volume_mixing_ratio[i] = typhon.physics.relative_humidity2vmr(RELATIVE_HUMIDITY, PRESSURE_LEVELS[i], t_profile[i])

        # handle tropopause
        if t_profile[i] <= COLD_POINT_TEMPERATURE_TROPOPAUSE and first_tp_p_lvl == True:

            # remember pressure level of tropopause
            pressure_level_tropopause = PRESSURE_LEVELS[i]

            # set wmr for tropopause
            water_vmr_tropopause = typhon.physics.relative_humidity2vmr(RELATIVE_HUMIDITY, pressure_level_tropopause, COLD_POINT_TEMPERATURE_TROPOPAUSE)
            first_tp_p_lvl = False

        # atmosphere profile if stratosphere is reached
        if t_profile[i] <= COLD_POINT_TEMPERATURE_TROPOPAUSE:

            # keep constant stratospheric temperature and wmr
            t_profile[i] = COLD_POINT_TEMPERATURE_TROPOPAUSE
            water_vapor_volume_mixing_ratio[i] =  water_vmr_tropopause

    return t_profile, water_vapor_volume_mixing_ratio, PRESSURE_LEVELS

def plot_profiles(t_profile, water_vapor_volume_mixing_ratio):
    heights = typhon.physics.pressure2height(PRESSURE_LEVELS) / 1000  # km
    pressures_hpa = PRESSURE_LEVELS / 100                             # Pa → hPa

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # ---- Temperature Profile ----
    ax = axes[0]
    ax.plot(t_profile, pressures_hpa)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title("Temperature")
    ax.invert_yaxis()
    ax.grid(True)

    # ---- Pressure vs Height ----
    ax = axes[1]
    ax.plot(pressures_hpa, heights)
    ax.set_xlabel('Pressure (hPa)')
    ax.set_ylabel('Height (km)')
    ax.set_title("Pressure profile")
    ax.grid(True)

    # ---- Water Vapor Mixing Ratio ----
    ax = axes[2]
    ax.plot(water_vapor_volume_mixing_ratio, pressures_hpa)
    ax.set_xlabel('Mixing Ratio (Pa/Pa)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title("Water Vapor VMR")
    ax.invert_yaxis()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("vertical_profiles.pdf")
    plt.show()

def main():
    vertical_profile = create_vertical_profile(t_0)
    temp_profile = vertical_profile[0]
    wmr_profile = vertical_profile[1]

    # plot temp, wmr and pressure profile
    plot_profiles(temp_profile, wmr_profile)

if __name__ == "__main__":
    main()