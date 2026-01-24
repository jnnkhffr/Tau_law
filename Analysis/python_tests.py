import numpy as np
import typhon
import pyarts3 as pa
KAYSER_GRID = np.linspace(1, 2000, 4)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)

temp = [290, 300, 310, 197]


planck_rad = typhon.physics.planck(FREQ_GRID, np.array(temp))
print("planck_rad shape: ", planck_rad.shape)