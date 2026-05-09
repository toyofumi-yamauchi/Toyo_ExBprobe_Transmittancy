#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import _Ion_info as _Ion_info
classes = [name for name, obj in vars(_Ion_info).items()
           if isinstance(obj, type)]
ion = _Ion_info.Ar
ion_name = ion.name
m_ion    = ion.mass
q_ion    = ion.charge
print('Chosen ion: {}'.format(ion_name))

import _ExB_probe_info as _ExB_probe_info
myExBprobe = _ExB_probe_info.Sample_ExB_probe
print('Chosen ExB probe: {}'.format(myExBprobe))

from Transmittancy import _transmittancy_calc as _transmittancy_calc
g = _transmittancy_calc.geometry()
f = _transmittancy_calc.field()
p = _transmittancy_calc.particle()

G     = g._geometric_const_G(myExBprobe) # Geometric constant, 1/m
α_max = g._max_incident_angle(myExBprobe) # Max incident angle, deg
Bxpra = f._Bxpra(myExBprobe) # Practical B-field strength, T

#%
v_ion    = 11.5e3 # ion velocity, m/s
vwpra    = 11.5e3 # Wien velocity, m/s
α_x      =  0.0 # the ion incident angle, deg
α_y      =  0.0 # the ion incident angle, deg

from _transmittancy_calc import _sunflower_seed
N_points = 10
r_ion    = 0.99*myExBprobe.r_1
x_0, y_0 = _sunflower_seed(N_points,r_ion) # m
z_0      = np.ones(N_points)*(-myExBprobe.l_f/2 - myExBprobe.l_c) # m

v_x_0    = np.ones(N_points)*v_ion*np.cos(α_y*np.pi/180)*np.sin(α_x*np.pi/180) # m/s
v_y_0    = np.ones(N_points)*v_ion*np.sin(α_y*np.pi/180) # m/s
v_z_0    = np.ones(N_points)*v_ion*np.cos(α_y*np.pi/180)*np.cos(α_x*np.pi/180) # m/s
r_0      = np.array((x_0,y_0,z_0,v_x_0,v_y_0,v_z_0))
t_max    = abs((myExBprobe.l_c+myExBprobe.l_f+myExBprobe.l_d)/v_ion)*2 # maximum time limit, s
t_gyro   = 2*np.pi*m_ion/(q_ion*myExBprobe.Bxpra) # gyro period, s
t_E      = m_ion*v_ion/(q_ion*np.max(f._E(myExBprobe,0,0,0,vwpra))) # E-field acceleration period, s
print('dt_max = {:.2e} s, dt_gyro = {:.2e} s'.format(t_max/100,t_gyro/25))
dt       = np.min([t_max/100, t_gyro/25]) # time step, s
t_array  = np.arange(0,t_max,dt) # time array, s

#%
r_n_array       = np.zeros((N_points,len(t_array),6))
n_last_array    = np.zeros( N_points                )
end_TF_array    = np.zeros( N_points                )
F_n_array       = np.zeros((N_points,len(t_array),7))

count_lost      = 0
count_collected = 0
n_collected = []
start = datetime.now()
for n in range(0,N_points):

    print(' Iteration: {}/{}'.format(n+1,N_points), end=" ")

    # print('Iteration: {}/{}'.format(n+1,N_points))
    # print(' (x_0, y_0, z_0) = ({:.2f}, {:.2f}, {:.2f}) mm'.format(1e3*x_0[n],1e3*y_0[n],1e3*z_0[n]))
    r_0 = np.array((x_0[n],y_0[n],z_0[n],v_x_0[n],v_y_0[n],v_z_0[n]))
    
    # Initial half push-back & Redefine the initial condition
    r_0, n_last_0, F_0, c_lost, c_collected = p.boris_bunemann_3D(myExBprobe,ion,vwpra,r_0,t_array[:2],output_TF=False)
    r_0 = np.array((x_0[n],y_0[n],z_0[n],r_0[1,3],r_0[1,4],r_0[1,5]))

    # Partile push through a time array
    r_n_array[n,:,:], n_last_array[n], F_n_array[n,:,:], c_lost, c_collected = p.boris_bunemann_3D(myExBprobe,ion,vwpra,r_0,t_array,output_TF=True)
    count_lost      += c_lost
    count_collected += c_collected
    if c_collected == 1:
       n_collected.append(n)
       end_TF_array[n] = 1

end = datetime.now()
print('Total computation time : {} s'.format((end - start).total_seconds()))
print('Average computation time: {} s'.format((end - start).total_seconds()/N_points))
print('Lost     : {} / {}'.format(count_lost ,N_points))
print('Collected: {} / {}'.format(count_collected,N_points))

import pyvista as pv
pv.close_all()
pv.set_jupyter_backend("static")

def make_polylines_subset(trajectory_data,n_last_array,select_mask,last_is_index=True):
    """
    Build a PyVista PolyData with only the trajectories selected by select_mask.

    Parameters
    ----------
    trajectory_data : array, shape (N_traj, N_step, 6)
        Trajectory data: x, y, z, vx, vy, vz
    n_last_array : array, shape (N_traj,)
        Last valid step index (or number of valid steps)
    select_mask : boolean array, shape (N_traj,)
        True for trajectories to include
    last_is_index : bool
        If True, n_last_array[i] is the last valid index.
        If False, n_last_array[i] is the number of valid steps.
    """

    trajectory_data = np.asarray(trajectory_data, dtype=float)
    n_last_array = np.asarray(n_last_array, dtype=int)
    select_mask = np.asarray(select_mask, dtype=bool)

    n_traj, n_step, n_components = trajectory_data.shape

    all_points = []
    all_lines = []
    all_traj_id = []

    point_offset = 0

    for i in range(n_traj):
        if not select_mask[i]:
            continue

        if last_is_index:
            n_valid = int(n_last_array[i]) + 1
        else:
            n_valid = int(n_last_array[i])

        n_valid = max(0, min(n_valid, n_step))

        if n_valid < 2:
            continue

        points = trajectory_data[i, :n_valid, :3]

        # optional cleanup in case of NaN/inf
        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]

        if len(points) < 2:
            continue

        n_pts = len(points)

        all_points.append(points)

        ids = np.arange(point_offset, point_offset + n_pts)
        line = np.hstack(([n_pts], ids))
        all_lines.append(line)

        all_traj_id.append(np.full(n_pts, i))

        point_offset += n_pts

    if len(all_points) == 0:
        return None

    poly = pv.PolyData()
    poly.points = np.vstack(all_points)
    poly.lines = np.concatenate(all_lines).astype(np.int64)
    poly.point_data["trajectory_id"] = np.concatenate(all_traj_id)

    return poly

body_mesh    = pv.read(myExBprobe.directory+myExBprobe.filename_CAD)
coll_mesh    = pv.read(myExBprobe.directory+myExBprobe.filename_CAD_collector)
body_surface = body_mesh.extract_surface(algorithm='dataset_surface').triangulate().clean()
coll_surface = coll_mesh.extract_surface(algorithm='dataset_surface').triangulate().clean()
clip_origin  = body_surface.center
try:
    # Best for closed surface geometry, e.g. STL shells.
    clipped_body_surface = body_surface.clip_closed_surface(normal="-x",origin=clip_origin)
    clipped_coll_surface = coll_surface.clip_closed_surface(normal="-x",origin=clip_origin)
except Exception:
    # Fallback for non-closed or difficult meshes.
    clipped_body_surface = body_surface.clip(normal="-x",origin=clip_origin,invert=False)
    clipped_coll_surface = coll_surface.clip(normal="-x",origin=clip_origin,invert=False)

red_mask = (end_TF_array == 1)
blue_mask = (end_TF_array == 0)
trajectory_data  = r_n_array/0.0254  # shape: (N_points, N_step, 6)
red_lines  = make_polylines_subset(trajectory_data,n_last_array,red_mask ,last_is_index=True)
blue_lines = make_polylines_subset(trajectory_data,n_last_array,blue_mask,last_is_index=True)
red_tubes  = None if red_lines  is None else red_lines .tube(radius=myExBprobe.r_1*25.4/10, n_sides=12)
blue_tubes = None if blue_lines is None else blue_lines.tube(radius=myExBprobe.r_1*25.4/10, n_sides=12)

p = pv.Plotter(window_size=(800, 600))
p.add_mesh(                           clipped_body_surface,color="lightgray",opacity=1.00,smooth_shading=False,label="Body")
p.add_mesh(                           clipped_coll_surface,color="gold"     ,opacity=1.00,smooth_shading=False,label="Collector")
if red_tubes is not None:  p.add_mesh(red_tubes           ,color="red"                   ,smooth_shading=True ,label="Collected ions")
if blue_tubes is not None: p.add_mesh(blue_tubes          ,color="blue"                  ,smooth_shading=True ,label="Lost ions")
p.camera_position = "yz" # view the XZ plane
p.camera.roll      =  180 # rotate image in-plane, optional
p.camera.azimuth   =   0
p.camera.elevation =   0
p.enable_parallel_projection()
p.camera.zoom(1.1)         # zoom in, optional
p.add_axes()
p.show_bounds(grid="front", location="outer", xtitle="X [in]", ytitle="Y [in]", ztitle="Z [in]")
p.add_legend(bcolor='gray',border=True,loc='lower right')
p.show(screenshot=myExBprobe.directory+"ion_trajectory_geometry.png")
p.export_html(myExBprobe.directory+"ion_trajectory_geometry.html")

