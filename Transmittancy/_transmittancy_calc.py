from functools import lru_cache
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
import trimesh
from pathlib import Path
from typing import Dict, Tuple

def _sunflower_seed(n,r):
    # stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
    k_theta = np.pi*(3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta*n, n)
    r = np.sqrt(np.linspace(0,r**2,n))
    return r*np.stack((np.cos(angles),np.sin(angles)))

class particle:
    def __init__(self, field_obj=None, geom_obj=None):
        self._cache = {}
        self.field = field_obj or field()
        self.geom  = geom_obj  or geometry()

    #% Numerical transmittancy calculation
    def _transmittancy_calculation(self, myExBprobe: object, ion: object, v_w: float, r_initials: np.ndarray, t_array: np.ndarray, BC_result_show_TF: bool) -> Tuple[int, int]:
        count_lost = 0
        count_collected = 0
        
        N_points = np.size(r_initials,1)
        for i in range(0,N_points):
            if BC_result_show_TF == True: print(' Iteration: {}/{}'.format(i+1,N_points), end=" ")
            # print(' (x_0, y_0, z_0) = ({:.2f}, {:.2f}, {:.2f}) mm'.format(x_0[i]*1e3,y_0[i]*1e3,z_0[i]*1e3))
            x_0, y_0, z_0, v_x_0, v_y_0, v_z_0 = r_initials[:,i]
            # Initial half push-back & Redefine the initial condition
            r_0, n_last_0, F_0, c_lost, c_collected = self.boris_bunemann_3D(myExBprobe,ion,v_w,r_initials[:,i],-t_array[:2], output_TF=False)
            r_initials[:,i] = np.array((x_0,y_0,z_0,r_0[1,3],r_0[1,4],r_0[1,5]))
            # Partile push through a time array
            r_n, n_last,   F_n, c_lost, c_collected = self.boris_bunemann_3D(myExBprobe,ion,v_w,r_initials[:,i], t_array    , output_TF=BC_result_show_TF)
            count_lost      += c_lost
            count_collected += c_collected
        if count_collected + count_lost != N_points: print(count_collected+count_lost)
        return count_lost, count_collected

    def boris_bunemann_3D(self, myExBprobe: object, ion: object, v_w: float, r_0: np.ndarray, time: np.ndarray, output_TF: bool) -> Tuple[np.ndarray, int, np.ndarray, int, int]:
        '''
        Input:
        myExBprobe        = class containing the ExB probe information
        ion               = class containing the ion information
        v_w               = Wien velocity, m/s
        time              = 1D time array, s
        r_0               = [x_0,y_0,z_0,vx_0,vy_0,vz_0] = initial condition, m and m/s
        Output:
        r_vec             = [x_n,y_n,z_n,vx_n,vy_n,vz_n] x # of time steps (6 x n), m and m/s
        n_last            = the # of iterations, -
        F_vec             = [Bx_n,By_n,Bz_n,Ex_n,Ey_n,Ez_n,phi_n] x # of time steps (7 x n), T and V/m and V
        counter_lost      = 1 if the particle is lost, 0 if not, -
        counter_collected = 1 if the particle is collected, 0 if not, -
        '''
        # start = datetime.now()
        
        scaling_factor = v_w*myExBprobe.d_e*myExBprobe.Bxpra/myExBprobe.V_electrodes

        dt = time[1] - time[0]             # time increment, s
        qmdt2 = 0.5*ion.charge/ion.mass*dt # magic constant, C*s/kg
        n_max = np.size(time)              # the max # of iterations, -
        r_n = np.zeros((n_max,6))          # setting the r_n array
        F_n = np.zeros((n_max,7))          # setting the F_n array
        r_n[0,:] = r_0                     # assigning the initial condition to r_n array
        x,y,z,vx,vy,vz = r_0

        for n in range(0,n_max-1):

            # Ex,Ey,Ez = _Efield_3D(point)
            Ex,Ey,Ez,phi = self.field._E(myExBprobe, x, y, z, scaling_factor)
            Bx,By,Bz     = self.field._B(myExBprobe, x, y, z, myExBprobe.Bx0_measured/myExBprobe.Bx0_simulated)

            alpha_x = self.frequency_correction(qmdt2*Bx)
            alpha_y = self.frequency_correction(qmdt2*By)
            alpha_z = self.frequency_correction(qmdt2*Bz)

            v_minus_x = vx + qmdt2*Ex*alpha_x
            v_minus_y = vy + qmdt2*Ey*alpha_y
            v_minus_z = vz + qmdt2*Ez*alpha_z
            v_minus_vec = np.array((v_minus_x,v_minus_y,v_minus_z))

            t_x = qmdt2*Bx*alpha_x
            t_y = qmdt2*By*alpha_y
            t_z = qmdt2*Bz*alpha_z
            t_mag = np.sqrt(t_x**2 + t_y**2 + t_z**2)
            t_vec = np.array((t_x,t_y       ,t_z))

            s_x = 2.0*t_x/(1.0+t_mag**2)
            s_y = 2.0*t_y/(1.0+t_mag**2)
            s_z = 2.0*t_z/(1.0+t_mag**2)
            s_vec = np.array((s_x,s_y,s_z))

            v_prime_vec = v_minus_vec + np.cross(v_minus_vec,t_vec)
            v_plus_vec  = v_minus_vec + np.cross(v_prime_vec,s_vec)

            v_plus_x = v_plus_vec[0]
            v_plus_y = v_plus_vec[1]
            v_plus_z = v_plus_vec[2]

            vx = v_plus_x + qmdt2*Ex*alpha_x
            vy = v_plus_y + qmdt2*Ey*alpha_y
            vz = v_plus_z + qmdt2*Ez*alpha_z

            x = x + vx*dt
            y = y + vy*dt
            z = z + vz*dt

            r_n[n+1,:] = np.array((x,y,z,vx,vy,vz))
            F_n[n+1,:] = np.array((Bx,By,Bz,Ex,Ey,Ez,phi))

            n_last = int(n+1)

            BC_test, counter_lost, counter_collected = self.geom._BC(myExBprobe,x,y,z,output_TF)
            if BC_test == True:
                # print(' - Particle out of the boundary at t = {:.2e} s'.format(n*dt))
                break

        # end = datetime.now()
        # print('Computation time: {}'.format(end - start))

        return r_n, n_last, F_n, counter_lost, counter_collected
    
    def frequency_correction(self, x):
        # Input: x = Ω*dt/2 = (q/m*dt/2)*B, s
        # Output: alpha = frequency correction factor
        if x == 0:
            alpha = 1.0
        else:
            alpha = np.tan(x)/x
        return alpha

    #% Analytical transmittancy calculation
    def _transmittancy_calculation_analitycal(self, myExBprobe: object, ion: object, v_w: float, v_ion: float, alpha_x: float, alpha_y: float) -> float:
        v_x_0 = v_ion*np.cos(alpha_y*np.pi/180)*np.sin(alpha_x*np.pi/180) # m/s
        v_y_0 = v_ion*np.sin(alpha_y*np.pi/180) # m/s
        v_z_0 = v_ion*np.cos(alpha_y*np.pi/180)*np.cos(alpha_x*np.pi/180) # m/s

        Bxpra = myExBprobe.Bxpra

        Δx_c = myExBprobe.l_c*np.tan(alpha_x*np.pi/180) # Δx in collimator, m
        Δx_f = myExBprobe.l_f*np.tan(alpha_x*np.pi/180) # Δx in ExB region, m
        Δx_d = myExBprobe.l_d*np.tan(alpha_x*np.pi/180) # Δx in drift tube, m

        Δy_c = myExBprobe.l_c*np.tan(alpha_y*np.pi/180)                                                                                    # Δy in collimator, m
        Δy_f = myExBprobe.l_f*np.tan(alpha_y*np.pi/180) + 0.5*ion.charge/ion.mass*Bxpra*myExBprobe.l_f*myExBprobe.l_f*(v_z_0-v_w)/v_z_0**2 # Δy in ExB region, m
        Δy_d = myExBprobe.l_d*np.tan(alpha_y*np.pi/180) + 1.0*ion.charge/ion.mass*Bxpra*myExBprobe.l_f*myExBprobe.l_d*(v_z_0-v_w)/v_z_0**2 # Δy in drift tube, m

        Δr_c = np.sqrt(Δx_c**2 + Δy_c**2) # Δr in collimator, m
        Δr_f = np.sqrt(Δx_f**2 + Δy_f**2) # Δr in ExB region, m
        Δr_d = np.sqrt(Δx_d**2 + Δy_d**2) # Δr in drift tube, m

        A_1 = np.pi*myExBprobe.r_1**2 # cross-sectional area at 1, m^2
        A_2 = np.pi*myExBprobe.r_2**2 # cross-sectional area at 2, m^2
        A_3 = np.pi*myExBprobe.r_3**2 # cross-sectional area at 3, m^2
        A_4 = np.pi*myExBprobe.r_4**2 # cross-sectional area at 4, m^2
        A_min = min(A_1,A_2,A_3,A_4)     

        S_12 = self.intersection_two_circles(myExBprobe.r_1, myExBprobe.r_2, Δr_c)           # intersection area of 1 and 2, m^2
        S_13 = self.intersection_two_circles(myExBprobe.r_1, myExBprobe.r_3, Δr_f)           # intersection area of 1 and 3, m^2
        S_14 = self.intersection_two_circles(myExBprobe.r_1, myExBprobe.r_4, Δr_d)           # intersection area of 1 and 4, m^2
        S_23 = self.intersection_two_circles(myExBprobe.r_2, myExBprobe.r_3, abs(Δr_c-Δr_f)) # intersection area of 2 and 3, m^2
        S_24 = self.intersection_two_circles(myExBprobe.r_2, myExBprobe.r_4, abs(Δr_c-Δr_d)) # intersection area of 2 and 4, m^2
        S_34 = self.intersection_two_circles(myExBprobe.r_3, myExBprobe.r_4, abs(Δr_f-Δr_d)) # intersection area of 3 and 4, m^2

        if min((S_12, S_13, S_14, S_23, S_24, S_34)) == 0:
            T = 0
        elif all((S_12, S_13, S_14)) == A_1 and all(S_23, S_24) == A_2 and S_34 == A_3:
            T = 1
        else:
            T = min(S_12, S_13, S_14, S_23, S_24, S_34)/A_min

        return T

    def intersection_two_circles(self,r_i,r_j,Δ_y):
        # tjkendev.github.io/procon-library/python/geometry/circlesintersection_area.html
        # Input:
        # r_i  = the radius i, m
        # r_j  = the radius j, m
        # Δ_y  = the separation distance b/w two circles, m
        # Output:
        # S_ij = the intersection area of two ciecles, m^2

        if abs(r_i + r_j)**2 <= Δ_y**2:
            S_ij = 0
        elif abs(r_i - r_j)**2 >= Δ_y**2:
            S_ij = np.pi*min(r_i,r_j)**2
        else:
            S_ij = r_i**2*np.arctan2(np.sqrt(4*r_i**2*Δ_y**2-(r_i**2-r_j**2+Δ_y**2)**2),(r_i**2-r_j**2+Δ_y**2))\
                  +r_j**2*np.arctan2(np.sqrt(4*r_j**2*Δ_y**2-(r_j**2-r_i**2+Δ_y**2)**2),(r_j**2-r_i**2+Δ_y**2))\
                                -0.5*np.sqrt(4*r_i**2*Δ_y**2-(r_i**2-r_j**2+Δ_y**2)**2)

        return S_ij


class geometry:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
    
    def clear_cache(self, filename: str = None) -> None:
        """Clear the entire cache or a single file."""
        if filename is None:
            self._cache.clear()
        else:
            self._cache.pop(str(filename), None)

    #% Boundary condition for numerically calculate transmittancy matrix
    def _BC(self, myExBprobe, x, y, z, message_TF=True):

        filename_body = str(myExBprobe.directory + myExBprobe.filename_CAD)
        source_units = myExBprobe.source_units
        entry_body = self._ensure_mesh_loaded(filename_body,source_units)
        mesh_body = entry_body['mesh']

        filename_collector = str(myExBprobe.directory + myExBprobe.filename_CAD_collector)
        source_units = myExBprobe.source_units
        entry_collector = self._ensure_mesh_loaded(filename_collector,source_units)
        mesh_collector = entry_collector['mesh']

        r = np.hypot(x, y)
        inside_mesh_body_TF      = mesh_body.contains([(x, y, z)])[0]
        inside_mesh_collector_TF = mesh_collector.contains([(x, y, z)])[0]

        TF = False
        coun_lost = 0
        coun_collected = 0

        if inside_mesh_body_TF == True:
            coun_lost = 1
            TF = True
            message = ' - Ion hit wall'
            if message_TF: print(message)

        elif inside_mesh_collector_TF == True:
            coun_collected = 1
            TF = True
            message = ' - Ion hit collector'
            if message_TF: print(message)

        # elif z < -(myExBprobe.l_f/2 + myExBprobe.l_c):
        #     TF = True
        #     coun_lost = 1
        #     message = ' - Ion escaped from the 1st aperture'
        #     if message_TF: print(message)

        return TF, coun_lost, coun_collected
    
    def _ensure_mesh_loaded(self, filename: str, source_unit: str) -> dict:
        
        if filename not in self._cache:
            self.preload_mesh(filename, source_units = source_unit)
        return self._cache[filename]

    def preload_mesh(self, filename: str, source_units: str = "inch", target_units: str = "m") -> None:
        """Load and cache the scaled mesh for later use."""
        filename = str(filename)
        if filename in self._cache:
            return  # already cached
        
        print(f"Loading mesh from file: {filename}")
        mesh = trimesh.load_mesh(filename)
        scale = 1.0
        if source_units == "mm" and target_units == "m":
            scale = 1.0/1000.0
        elif source_units == "cm" and target_units == "m":
            scale = 1.0/100.0
        elif source_units == "inch" and target_units == "m":
            scale = 0.0254
        if scale != 1.0:
            mesh.apply_scale(scale)

        # print(' x_min, x_max = {:.3f}, {:.3f} m'.format(mesh.bounds[0,0],mesh.bounds[1,0]))
        # print(' y_min, y_max = {:.3f}, {:.3f} m'.format(mesh.bounds[0,1],mesh.bounds[1,1]))
        # print(' z_min, z_max = {:.3f}, {:.3f} m'.format(mesh.bounds[0,2],mesh.bounds[1,2]))
        
        self._cache[filename] = dict(
            mesh=mesh,
            mtime=Path(filename).stat().st_mtime,
        )

    #% Typical ExB probe geometry parameters
    def _geometric_const_G(self, myExBprobe: object) -> float:
        if type(myExBprobe.l_d) == list:
            G = []
            for i in range(0,len(myExBprobe.l_d)):
                G.append((myExBprobe.r_2 + myExBprobe.r_3 + (myExBprobe.r_1 + myExBprobe.r_2)*(myExBprobe.l_f + myExBprobe.l_d[i])/myExBprobe.l_c)/(myExBprobe.l_f**2 + 2*myExBprobe.l_d[i]*myExBprobe.l_f))
        else:
            G = (myExBprobe.r_2 + myExBprobe.r_3 + (myExBprobe.r_1 + myExBprobe.r_2)*(myExBprobe.l_f + myExBprobe.l_d)/myExBprobe.l_c)/(myExBprobe.l_f**2 + 2*myExBprobe.l_d*myExBprobe.l_f)
        return G
    
    def _max_incident_angle(self, myExBprobe: object) -> float:
        α_max = np.arctan2(myExBprobe.r_1 + myExBprobe.r_2, myExBprobe.l_c)*180/np.pi
        return α_max

class field:
    def __init__(self):
        self._cache: Dict[str, dict] = {}

    def clear_cache(self, filename: str = None) -> None:
        """Clear the entire cache or entries derived from a file."""
        if filename is None:
            self._cache.clear()
            return
        # Remove loaded field/potential entries
        self._cache.pop(str(filename), None)
        # Also drop any Bxpra entries that reference this filename
        bx = self._cache.get('__bxpra__')
        if bx:
            to_del = [k for k in bx.keys() if (isinstance(k, tuple) and str(filename) in k[:3])]
            for k in to_del:
                bx.pop(k, None)

    def _B(self, myExBprobe: object, x, y, z, scale: float = 1.0):
        """
        Interpolate B at (x, y, z). Supports scalars or arrays (broadcasted).
        Returns shape (..., 3) in Tesla or V/m after scaling.
        """

        filename = str(myExBprobe.directory + myExBprobe.filename_B)

        entry = self._ensure_Bfield_loaded(filename)
        Bx_i, By_i, Bz_i = entry['Fx_interp'], entry['Fy_interp'], entry['Fz_interp']

        # Broadcast inputs to common shape, then query as (N,3)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        x, y, z = np.broadcast_arrays(x, y, z)
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

        Bx = Bx_i(pts)
        By = By_i(pts)
        Bz = Bz_i(pts)

        scale = myExBprobe.Bx0_measured/myExBprobe.Bx0_simulated

        B = np.stack([Bx, By, Bz], axis=-1).reshape(x.shape + (3,))
        if scale != 1.0:
            B = B * scale
        return B

    def _ensure_Bfield_loaded(self, filename: str) -> dict:
        filename = str(filename)
        if filename not in self._cache:
            self.preload_Bfield(filename)
        return self._cache[filename]

    def preload_Bfield(self, filename: str) -> None:
        """Load and cache the Field interpolators for later use."""
        filename = str(filename)
        if filename in self._cache:
            return  # already cached

        print(f"Loading field from file: {filename}")
        if filename[-4:] == '.fld':
            # print("Detected ANSYS .fld format")
            F_field = np.loadtxt(filename, dtype=float, skiprows=2)
        
        if filename[-4:] == '.txt':
            # print("Detected COMSOL .txt format")
            F_field = np.loadtxt(filename, dtype=float, skiprows=9, delimiter=',')

        # Build grids — assumes columns: x, y, z, Bx, By, Bz
        x_vals = np.unique(np.round(F_field[:, 0],5)) # m
        y_vals = np.unique(np.round(F_field[:, 1],5)) # m
        z_vals = np.unique(np.round(F_field[:, 2],5)) # m
        nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

        # print(' x_min, x_max = {:.4f}, {:.4f} m'.format(np.min(x_vals), np.max(x_vals)))
        # print(' y_min, y_max = {:.4f}, {:.4f} m'.format(np.min(y_vals), np.max(y_vals)))
        # print(' z_min, z_max = {:.3f}, {:.3f} m'.format(np.min(z_vals), np.max(z_vals)))
        # print(' (Nx, Ny, Nz) = ({:.0f}, {:.0f}, {:.0f})'.format(nx, ny, nz))
        
        if filename[-4:] == '.fld':
            Fx_array = F_field[:, 3].reshape(nx, ny, nz)
            Fy_array = F_field[:, 4].reshape(nx, ny, nz)
            Fz_array = F_field[:, 5].reshape(nx, ny, nz)

        if filename[-4:] == '.txt':
            x_map = {val: i for i, val in enumerate(x_vals)}
            y_map = {val: i for i, val in enumerate(y_vals)}
            z_map = {val: i for i, val in enumerate(z_vals)}
            
            Fx_array = np.zeros((nx, ny, nz))
            Fy_array = np.zeros((nx, ny, nz))
            Fz_array = np.zeros((nx, ny, nz))

            for row in F_field:
                # Find the integer indices (i, j, k) for this point
                i = x_map[np.round(row[0],4)]
                j = y_map[np.round(row[1],4)]
                k = z_map[np.round(row[2],4)]
                
                # Place the B-field values into the 3D arrays
                Fx_array[i, j, k] = row[3]
                Fy_array[i, j, k] = row[4]
                Fz_array[i, j, k] = row[5]

        # Create interpolators (store unscaled; apply scale at query time)
        Fx_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fx_array, method='linear', bounds_error=False, fill_value=None)
        Fy_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fy_array, method='linear', bounds_error=False, fill_value=None)
        Fz_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fz_array, method='linear', bounds_error=False, fill_value=None)

        self._cache[filename] = dict(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
            Fx_interp=Fx_interp, Fy_interp=Fy_interp, Fz_interp=Fz_interp,
            mtime=Path(filename).stat().st_mtime,
        )

    def _E(self, myExBprobe: object, x, y, z, scale: float = 1.0):
        """
        Interpolate E at (x, y, z). Supports scalars or arrays (broadcasted).
        Returns shape (..., 3) in Tesla or V/m after scaling.
        """

        filename = str(myExBprobe.directory + myExBprobe.filename_E)

        entry = self._ensure_Efield_loaded(filename,myExBprobe)
        Ex_i, Ey_i, Ez_i, Po_i = entry['Fx_interp'], entry['Fy_interp'], entry['Fz_interp'], entry['Po_interp']

        # Broadcast inputs to common shape, then query as (N,3)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        x, y, z = np.broadcast_arrays(x, y, z)
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

        Ex = Ex_i(pts)
        Ey = Ey_i(pts)
        Ez = Ez_i(pts)
        Po = Po_i(pts)

        E = np.stack([Ex, Ey, Ez, Po], axis=-1).reshape(x.shape + (4,))
        if scale != 1.0:
            E = E * scale
        return E

    def _ensure_Efield_loaded(self, filename: str, myExBprobe: object) -> dict:
        filename = str(filename)
        if filename not in self._cache:
            self.preload_Efield(filename,myExBprobe)
        return self._cache[filename]

    def preload_Efield(self, filename: str, myExBprobe: object) -> None:
        """Load and cache the Field interpolators for later use."""
        filename = str(filename)
        if filename in self._cache:
            return  # already cached

        print(f"Loading field from file: {filename}")
        if filename[-4:] == '.fld':
            # This is for ANSYS output
            F_field = np.loadtxt(filename, dtype=float, skiprows=2)
        if filename[-4:] == '.txt':
            # This is for COMSOL output
            F_field = np.loadtxt(filename, dtype=float, skiprows=9, delimiter=',')

        # Build grids — assumes columns: x, y, z, Ex, Ey, Ez, P
        x_vals = np.unique(np.round(F_field[:, 0],5)) # m
        y_vals = np.unique(np.round(F_field[:, 1],5)) # m
        z_vals = np.unique(np.round(F_field[:, 2],5)) # m
        nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

        # print(' x_min, x_max = {:.4f}, {:.4f} m'.format(np.min(x_vals), np.max(x_vals)))
        # print(' y_min, y_max = {:.4f}, {:.4f} m'.format(np.min(y_vals), np.max(y_vals)))
        # print(' z_min, z_max = {:.3f}, {:.3f} m'.format(np.min(z_vals), np.max(z_vals)))
        # print(' (Nx, Ny, Nz) = ({:.0f}, {:.0f}, {:.0f})'.format(nx, ny, nz))
        
        if filename[-4:] == '.fld':
            Fx_array = F_field[:, 3].reshape(nx, ny, nz)
            Fy_array = F_field[:, 4].reshape(nx, ny, nz)
            Fz_array = F_field[:, 5].reshape(nx, ny, nz)

            entry2 = self._ensure_potential_loaded(myExBprobe.directory+myExBprobe.filename_P)
            Po_array = entry2['P_interp'](np.array(np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')).reshape(3, -1).T).reshape(nx, ny, nz)


        if filename[-4:] == '.txt':
            x_map = {val: i for i, val in enumerate(x_vals)}
            y_map = {val: i for i, val in enumerate(y_vals)}
            z_map = {val: i for i, val in enumerate(z_vals)}
            
            Fx_array = np.zeros((nx, ny, nz))
            Fy_array = np.zeros((nx, ny, nz))
            Fz_array = np.zeros((nx, ny, nz))
            Po_array = np.zeros((nx, ny, nz))

            for row in F_field:
                # Find the integer indices (i, j, k) for this point
                i = x_map[np.round(row[0],4)]
                j = y_map[np.round(row[1],4)]
                k = z_map[np.round(row[2],4)]
                
                # Place the B-field values into the 3D arrays
                Fx_array[i, j, k] = row[3]
                Fy_array[i, j, k] = row[4]
                Fz_array[i, j, k] = row[5]
                Po_array[i, j, k] = row[6]

        # Create interpolators (store unscaled; apply scale at query time)
        Fx_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fx_array, method='linear', bounds_error=False, fill_value=None)
        Fy_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fy_array, method='linear', bounds_error=False, fill_value=None)
        Fz_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Fz_array, method='linear', bounds_error=False, fill_value=None)
        Po_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), Po_array, method='linear', bounds_error=False, fill_value=None)

        self._cache[filename] = dict(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
            Fx_interp=Fx_interp, Fy_interp=Fy_interp, Fz_interp=Fz_interp, Po_interp=Po_interp,
            mtime=Path(filename).stat().st_mtime,
        )


    def _ensure_potential_loaded(self, filename: str) -> dict:
        filename = str(filename)
        if filename not in self._cache:
            self.preload_potential(filename)
        return self._cache[filename]

    def preload_potential(self, filename: str) -> None:
        """Load and cache the Potential interpolators for later use."""
        filename = str(filename)
        if filename in self._cache:
            return  # already cached

        print(f"Loading potential from file: {filename}")
        P_field = np.loadtxt(filename, dtype=float, skiprows=2)

        # Build grids — assumes columns: x, y, z, Bx, By, Bz
        x_vals = np.unique(P_field[:, 0])
        y_vals = np.unique(P_field[:, 1])
        z_vals = np.unique(P_field[:, 2])
        nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

        # print(' x_min, x_max = {:.3f}, {:.3f} m'.format(np.min(x_vals), np.max(x_vals)))
        # print(' y_min, y_max = {:.3f}, {:.3f} m'.format(np.min(y_vals), np.max(y_vals)))
        # print(' z_min, z_max = {:.3f}, {:.3f} m'.format(np.min(z_vals), np.max(z_vals)))
        # print(' (Nx, Ny, Nz) = ({:.0f}, {:.0f}, {:.0f})'.format(nx, ny, nz))
        
        # Reshape into 3D arrays (UNSCALED Tesla)
        P_array = P_field[:, 3].reshape(nx, ny, nz)
        
        # Create interpolators (store unscaled; apply scale at query time)
        P_interp = RegularGridInterpolator((x_vals, y_vals, z_vals), P_array, method='linear', bounds_error=False, fill_value=None)

        self._cache[filename] = dict(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
            P_interp=P_interp,
            mtime=Path(filename).stat().st_mtime,
        )
    
    #% Practical B-field
    def _Bxpra(self, myExBprobe, *, N: int = 1000, force: bool = False, verbose: bool = False) -> float:
        """
        Return practical B-field (Bxpra). Cached by files, probe params, mtimes, and N.
        """
        # one bucket under the main cache
        bx_cache = self._cache.setdefault('__bxpra__', {})

        # build the cache key inline (same tuple wherever we need it)
        bfile = str(myExBprobe.directory + myExBprobe.filename_B)
        efile = str(myExBprobe.directory + myExBprobe.filename_E)
        # pfile = str(myExBprobe.directory + myExBprobe.filename_P)
        key = (
            bfile, efile,
            float(getattr(myExBprobe, "l_f", 0.0)),
            float(getattr(myExBprobe, "d_e", 0.0)),
            float(getattr(myExBprobe, "Bx0_measured", 1.0)),
            float(getattr(myExBprobe, "Bx0_simulated", 1.0)),
            int(N),
            Path(bfile).stat().st_mtime if Path(bfile).exists() else 0.0,
            Path(efile).stat().st_mtime if Path(efile).exists() else 0.0,
            # Path(pfile).stat().st_mtime if Path(pfile).exists() else 0.0,
        )

        if (not force) and (key in bx_cache):
            return bx_cache[key]

        val, _ = self.calculate_Bxpra(myExBprobe, N=N)
        bx_cache[key] = val
        return val

    def ensure_Bxpra_calculated(self, myExBprobe: object, *, N: int = 1000) -> float:
        """
        Return cached Bxpra if available; otherwise compute once and cache it.
        """
        bx_cache = self._cache.setdefault('__bxpra__', {})

        bfile = str(myExBprobe.directory + myExBprobe.filename_B)
        efile = str(myExBprobe.directory + myExBprobe.filename_E)
        # pfile = str(myExBprobe.directory + myExBprobe.filename_P)
        key = (
            bfile, efile,
            float(getattr(myExBprobe, "l_f", 0.0)),
            float(getattr(myExBprobe, "d_e", 0.0)),
            float(getattr(myExBprobe, "Bx0_measured", 1.0)),
            float(getattr(myExBprobe, "Bx0_simulated", 1.0)),
            int(N),
            Path(bfile).stat().st_mtime if Path(bfile).exists() else 0.0,
            Path(efile).stat().st_mtime if Path(efile).exists() else 0.0,
            # Path(pfile).stat().st_mtime if Path(pfile).exists() else 0.0,
        ).item()

        if key not in bx_cache:
            # compute via the same path as normal (populates the cache)
            self._Bxpra(myExBprobe, N=N, force=False, verbose=False)
        return bx_cache[key]

    def calculate_Bxpra(self, myExBprobe: object, *, N: int = 1000, verbose: bool = True):
        """
        Compute Bxpra and return (Bxpra, (Bxe, Eye)).
        Also writes the value into the keyed cache so future lookups hit.
        """
        # on-axis values
        B0 = self._B(myExBprobe, 0.0, 0.0, 0.0)
        E0 = self._E(myExBprobe, 0.0, 0.0, 0.0)
        # P0 = self._P(myExBprobe, 0.0, 0.0, 0.0)

        # Bx0, Ey0, Po0 = float(B0[0]), float(E0[1]), float(P0[0])
        Bx0, Ey0 = float(B0[0]), float(E0[1])

        # vectorized integration along z
        z = np.linspace(-myExBprobe.l_f/2, myExBprobe.l_f/2, N)
        x = y = np.zeros_like(z)

        B_line = self._B(myExBprobe, x, y, z)
        E_line = self._E(myExBprobe, x, y, z)

        Bxe = np.trapezoid(B_line[:, 0], z) / myExBprobe.l_f
        Eye = np.trapezoid(E_line[:, 1], z) / myExBprobe.l_f

        Bxpra = -Bxe * Ey0 / Eye

        if verbose:
            print('--- E- & B-field information ---')
            print(' B_x,0     = {:.1f} mT (< 0 mT)'.format(Bx0*1e3))
            print(' E_y,0     = {:.2f} V/m (> 0 V/m)'.format(Ey0))
            print(' E_y,0*d_e = {:.2f} V (≈ 10 V)'.format(Ey0*myExBprobe.d_e))
            # print(' P_0       = {:.2f} V (≈ 0 V)'.format(Po0))
            print(' B_x,e = {:.1f} mT, E_y,e = {:.2f} V/m'.format(Bxe*1e3, Eye))
            print(' Eff_B = {:.2f} %, Eff_E = {:.2f} %'.format(Bxe/Bx0*1e2, Eye/Ey0*1e2))
            print(' B_x,pra = {:.1f} mT'.format(Bxpra*1e3))
            print('-------------------------------')

        # return Bxpra, (Bx0, Ey0, Po0, Bxe, Eye)
        return Bxpra, (Bx0, Ey0, Bxe, Eye)