import matplotlib.pyplot as plt
import numpy as np
import pyarts3 as pyarts

# 1) Prepare ARTS workspace
pyarts.data.download()     # Download ARTS catalogs and set search path
ws = pyarts.workspace.Workspace()   # Initialize ARTS

# 2) Set up absorption species and read catalog data
ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400"])
ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")
ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

# 3) Set up line-by-line calculation
ws.lbl_checkedCalc()                # Check that the line-by-line data is consistent
ws.propmat_clearsky_agendaAuto()    # Set up propagation matrix calculation

# 4) Initialize required workspace variables
ws.stokes_dim = 1                   # Unpolarized
ws.jacobian_quantities = []         # No derivatives
ws.select_abs_species = []          # All species
ws.rtp_mag = []                     # No magnetic field
ws.rtp_los = []                     # No particular LOS
ws.rtp_nlte = pyarts.arts.EnergyLevelMap()   # No NLTE

# 5) Set up the frequency grid and the atmospheric conditions
f_grid_kayser = np.linspace(10, 2400, 30000)      # Frequency grid in Kayser
ws.f_grid = pyarts.arts.convert.kaycm2freq(f_grid_kayser)   # Convert to Hz
ws.rtp_pressure = 1000e2          # 1000 hPa
ws.rtp_temperature = 295          # At room temperature
ws.rtp_vmr = [0.02]               # H2O VMR

# 6) Calculate the absorption cross section
ws.AgendaExecute(a=ws.propmat_clearsky_agenda)    # Call the agenda
xsec = ws.propmat_clearsky.value.data.flatten() / (
    ws.rtp_vmr.value[0] * ws.rtp_pressure.value
    / (pyarts.arts.constants.k * ws.rtp_temperature.value)
) * 10000     # Convert absorption coefficients to cross sections in cm^2

# 7) Plot the absorption of this example
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(f_grid_kayser, xsec, lw=0.2, alpha=0.5, color="#932667")

def rolling_mean(x, w=1000):
    return np.convolve(x, np.ones(w) / w, "valid")

ax.semilogy(
    rolling_mean(f_grid_kayser),
    10 ** rolling_mean(np.log10(xsec)),
    lw=2,
    color="#932667"
)

ax.set_xlabel("Wavenumber / cm$^{-1}$")
ax.set_ylabel("Absorption cross section / cm$^2$ molecules$^{-1}$")
ax.set_xlim(f_grid_kayser.min(), f_grid_kayser.max())
ax.spines[["right", "top"]].set_visible(False)
fig.savefig("h2o-xsec.pdf")
plt.show()
