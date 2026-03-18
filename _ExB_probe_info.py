import numpy as np

class Sample_ExB_probe:
    def __init__(self): pass
    memo = "Sample"

    # Geometric parameters needed for analytical appraoch
    r_1 =   1.0e-3  # aperture 1 radius, m
    r_2 =   2.0e-3  # aperture 2 radius, m
    r_3 =   3.0e-3  # aperture 3 radius, m
    r_4 =   4.0e-3  # aperture 4 radius, m (= the collector radius)
    l_c =  50.0e-3  # collector length, m
    l_f = 100.0e-3  # filter length, m
    l_d =  50.0e-3  # drift tube length, m
    d_e =  10.0e-3  # electrode spacing, m
    
    # Parameters needed when visualizing trajectories
    z_electrodes = np.array((0.0e-3, 40.0e-3, 40.0e-3,  0.0e-3))
    y_electrodes = np.array((0.0e-3,  0.0e-3,  5.0e-3,  5.0e-3))+d_e/2
    y_electrodes = np.concatenate((y_electrodes, np.flip(y_electrodes)))
    z_electrodes = np.concatenate((z_electrodes,-np.flip(z_electrodes)))
    r_t          = 10e-3  # tube radius, m
    l_m          = 100e-3 # magnet length, m
    d_m          = 40e-3  # magnet separation distance, m

    # B-field and E-field parameters
    # You need to correct the magnitude of simulated B-field based on your measurement.
    Bx0_measured  = -0.20 # Actual measured B-field strength, Tesla
    Bx0_simulated =  0.15 # Simulated B-field strength, Tesla
    # Update this value by running _transmittancy_calc.field._Bxpra() for your ExB probe design. 
    Bxpra         =  0.224 # Practical B-field strength for ExB probe, Tesla. For more information, see Toyofumi Yamauchi and Joshua L. Rovey. "Uncertainty and Data Analysis of ExB Probe including Field Non-Uniformity and Transmittancy," AIAA 2024-0688. AIAA SCITECH 2024 Forum. January 2024.
    # Instead of running multiple E-field simulations, you can use one simulated E-field by scaling it to desired V_electrodes.
    V_electrodes  = 10 # potential difference between the electrodes used in E-field simulation
    
    # File information for numerical approach
    directory              = "../Sample/"                     # directory of these files. The calculated transmittancy matrix will be saved in this directory.
    filename_B             = "Sample_ExB_probe_B.txt"         # B-field data file, which is used for numerical approach.
    filename_E             = "Sample_ExB_probe_E.txt"         # E-field data file, which is used for numerical approach.
    filename_CAD           = "Sample_ExB_probe_WholeBody.stl" # CAD file for the whole ExB probe, which is used for numerical approach to determine if the ion hits the probe body or not.
    filename_CAD_collector = "Sample_ExB_probe_Collector.stl" # CAD file for the collector, which is used for numerical approach to determine if the ion hits the collector or not.
    source_units           = 'inch' # the unit of the coordinates in CAD files. "mm", "cm", and "inch" need to be converted to "m".

# test.py
if __name__ == "__main__":
    import sys
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if isinstance( getattr(current_module, key), type ):
            print(key)