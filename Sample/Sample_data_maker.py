#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from matplotlib.dates import DateFormatter
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Arrow
import scipy as sp
from scipy import special
from scipy import optimize
from scipy.signal import convolve2d, peak_widths
from datetime import datetime

e     = 1.602176634E-19              # elementary charge [C]
q_e   = -e                           # elctron charge [C]
m_e   = 9.1093837015E-31             # electron mass [kg]
N_A   = 6.02214076E+23               # Avogadro's number [#]
k_B   = 1.380649E-23                 # Boltzmann constant [J/K]
R_gas = N_A*k_B                      # Universal gas constant [J/mol/K]
c     = 299792458                    # speed of light [m/s]
μ_0   = 4.0*np.pi*1.0000000005415e-7 # vacuum permeability [N/A^2]
ε_0   = 1/μ_0/c**2                   # vacuum permittivity [F/m][C^2/N/m^2]
h_p   = 6.62607015e-34               # Planck constant [Js]
σ = 2*np.pi**5*k_B**4/15/h_p**3/c**2 # Stephan-Boltzmann constant [W/m^2/K^4]
π = np.pi                            # pi
cmap = plt.get_cmap('tab10')
CL = np.array(([*mcolors.TABLEAU_COLORS]))

figure_size = (6,4)
#figure_size = (9.67,6.39)
#figure_size = (4.75,6.33)

def _normal_distribution(x, A, μ, σ):
    return A/(σ*np.sqrt(2*np.pi))*np.exp(-(x-μ)**2/(2*σ**2))

header = "# Date Start, 2025-11-13 15:00:00.000000\n" \
"# ExB probe type, Sample ExB probe\n" \
"# ExB probe info, 1 meter away from thruster\n" \
"# Test condition note, HET, m_dot = 10 sccm, Xe, V_dis = 200 V, I_dis = 1 A\n" \
"# Time, V_electrode_+ [V], V_electrode_- [V] I_ave [A], I_SD [A], N_sample [count], I_max [A], I_min [A]"


V_electrodes = np.arange(2,122,1)
V_electrode_p =  V_electrodes/2
V_electrode_m = -V_electrodes/2
noise_array   = 1e-9*np.random.rand(len(V_electrodes))
offset_array  = 1e-9*np.ones(len(V_electrodes))
I_1 = _normal_distribution(V_electrodes, 100e-9, 40, 5)
I_2 = _normal_distribution(V_electrodes,  60e-9, 56, 5)
I_collected = I_1 + I_2 + noise_array + offset_array
I_sdev = 0.15*I_collected
N_sample = 3*np.ones(len(V_electrodes))
I_max = I_collected + 3*I_sdev
I_min = I_collected - 3*I_sdev
Time_array = np.array([datetime(2025,11,13,15,0,0) + i*np.timedelta64(1,'s') for i in range(len(V_electrodes))])

data = np.array((Time_array, V_electrode_p, V_electrode_m, I_collected, I_sdev, N_sample, I_max, I_min)).T
np.savetxt('Sample_data.txt', data, header=header, comments='', fmt=['%s','%.6e','%.6e','%.6e','%.6e','%.6e','%.6e','%.6e'], delimiter=',')