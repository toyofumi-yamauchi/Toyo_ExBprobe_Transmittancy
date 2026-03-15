e = 1.602176634E-19 # elementary charge [C]
N_A = 6.02214076E+23 # Avogadro's number [#]

class N2:
    name = '$N_2^+$'
    mass = 14.0067*2/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class N:
    name = '$N^+$'
    mass = 14.0067/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C
    
class O2:
    name = '$O_2^+$'
    mass = 15.999*2/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class O:
    name = '$O^+$'
    mass = 15.999/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class Xe:
    name = '$Xe^{+}$'
    mass = 131.293/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class Xe2:
    name = '$Xe^{2+}$'
    mass = 131.293/1000/N_A # particle mass [kg]
    charge = 2*e # the ion charge, C

class Xe3:
    name = '$Xe^{3+}$'
    mass = 131.293/1000/N_A # particle mass [kg]
    charge = 3*e # the ion charge, C

class Kr:
    name = '$Kr^{+}$'
    mass = 83.798/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C  

class Kr2:  
    name = '$Kr^{2+}$'
    mass = 83.798/1000/N_A # particle mass [kg]
    charge = 2*e # the ion charge, C

class Kr3:
    name = '$Kr^{3+}$'
    mass = 83.798/1000/N_A # particle mass [kg]
    charge = 3*e # the ion charge, C

class Ar:
    name = '$Ar^{+}$'
    mass = 39.948/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class Ar2:
    name = '$Ar^{2+}$'
    mass = 39.948/1000/N_A # particle mass [kg]
    charge = 2*e # the ion charge, C

class Ar3:
    name = '$Ar^{3+}$'
    mass = 39.948/1000/N_A # particle mass [kg]
    charge = 3*e # the ion charge, C

class H2O:
    name = '$H_2O^{+}$'
    mass = (1.008*2+15.999)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class H:
    name = '$H^{+}$'
    mass = 1.008/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class H2:
    name = '$H_2^{+}$'
    mass = 1.008*2/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class OH:
    name = '$OH^{+}$'
    mass = (1.008+15.999)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class H3O:
    name = '$H_3O^{+}$'
    mass = (1.008*3+15.999)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class NO:
    name = '$NO^{+}$'
    mass = (14.007+15.999)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class CO2:
    name = '$CO_2^{+}$'
    mass = (12.011+15.999*2)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class CO:
    name = '$CO^{+}$'
    mass = (12.011+15.999)/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

class C:
    name = '$C^{+}$'
    mass = 12.011/1000/N_A # particle mass [kg]
    charge = e # the ion charge, C

if __name__ == "__main__":
    import sys
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if isinstance( getattr(current_module, key), type ):
            print(key)