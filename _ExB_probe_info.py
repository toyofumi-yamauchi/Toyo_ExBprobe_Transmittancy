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
    l_f = 150.0e-3  # filter length, m
    l_d = 100.0e-3  # drift tube length, m
    d_e =  10.0e-3  # electrode spacing, m
    
    # Size of electrodes when visualizing traectories
    z_electrodes = np.array((0.0e-3, 50.0e-3, 50.0e-3,  0.0e-3))
    y_electrodes = np.array((0.0e-3,  0.0e-3, 10.0e-3, 10.0e-3))+d_e/2
    y_electrodes = np.concatenate((y_electrodes, np.flip(y_electrodes)))
    z_electrodes = np.concatenate((z_electrodes,-np.flip(z_electrodes)))

    # B-field and E-field parameters
    Bx0_measured  = -0.20 # Tesla, Bx at (x,y,z) = (0,0,0) measured on YYYY/MM/DD
    Bx0_simulated =  0.15 # Tesla, Bx at (x,y,z) = (0,0,0) simulated
    Bxpra         =  0.19 # Practical B-field strength for ExB probe, Tesla. See 
    V_electrodes = 10 # potential difference between the electrodes used in E-field simulation
    
    # File information for numerical approach
    directory = "../Sample/"
    filename_B   = "Sample_ExB_probe_B.txt"
    filename_E   = "Sample_ExB_probe_E.txt"
    filename_CAD = "Sample_ExB_probe_Body.stl"
    filename_CAD_collector = "Sample_ExB_probe_Collector.stl"
    source_units = 'inch'


# test.py
if __name__ == "__main__":
    import sys
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if isinstance( getattr(current_module, key), type ):
            print(key)