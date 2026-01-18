import numpy as np
import typhon
NUMBER_OF_PRESSURE_LEVELS = 100
PRESSURE_LEVELS = np.logspace(3, 0, NUMBER_OF_PRESSURE_LEVELS)
heights = typhon.physics.pressure2height(PRESSURE_LEVELS)
print(heights)
tau_heights = np.array([1000, 2000, 3000, 22000,40000])
indices = np.argmin(np.abs(heights[:, np.newaxis] - tau_heights), axis=0)
print(indices)