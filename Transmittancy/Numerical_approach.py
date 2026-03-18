#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

import _transmittancy_calc as _transmittancy_calc
g = _transmittancy_calc.geometry()
f = _transmittancy_calc.field()
p = _transmittancy_calc.particle(field_obj=f, geom_obj=g)

G     = g._geometric_const_G(myExBprobe) # Geometric constant, 1/m
α_max = g._max_incident_angle(myExBprobe) # Max incident angle, deg
Bxpra = f._Bxpra(myExBprobe) # Practical B-field strength, T

N           = 60
M           = 60
v_min       = 1e3
v_max       = 120e3
beta        = 2.0
eta_v_ion   = np.linspace(0, 1, N)
eta_vwpra   = np.linspace(0, 1, M)
v_ion_array = np.flip(np.round((v_min - v_max)*(np.tanh(beta*eta_v_ion) / np.tanh(beta)) + v_max,0))
vwpra_array = np.flip(np.round((v_min - v_max)*(np.tanh(beta*eta_vwpra) / np.tanh(beta)) + v_max,0))
Δv_w_array  = m_ion/q_ion*G*vwpra_array**2
α_x         =  0.0 # the ion incident angle, deg
α_y         =  0.0 # the ion incident angle, deg
print(v_ion_array)

XX, YY = np.meshgrid(vwpra_array/1e3, v_ion_array/1e3)
plt.figure(figsize=(7, 7))
# Plot the grid lines (horizontal and vertical lines)
for i in range(M-1): plt.plot((XX[0,i]+XX[0,i+1])/2*np.ones(N),YY[:, 0], 'k-', linewidth=0.5)
for j in range(N-1): plt.plot(XX[0,:],(YY[j,:]+YY[j+1,:])/2, 'k-', linewidth=0.5)
plt.scatter(XX, YY, color='k', s=20*XX*YY/((v_max-v_min)/1000)**2) # Plot nodes
plt.title(f'Graded Quadrilateral Mesh (tanh, β={beta})')
plt.xlabel('$v_w$, km/s')
plt.ylabel('v_ion (m/s)')
plt.grid(True)
plt.axis('equal')
plt.show()

from _transmittancy_calc import _sunflower_seed
N_points    = 10
r_ion       = 0.99*myExBprobe.r_1
x_0, y_0    = _sunflower_seed(N_points,r_ion)
z_0         = np.ones(N_points)*(-myExBprobe.l_f/2 - myExBprobe.l_c)
#%%
count_lost      = np.zeros((M,N))
count_collected = np.zeros((M,N))

z_check = -myExBprobe.l_f*0.4
if f._E(myExBprobe,0,0,z_check)[1]/f._E(myExBprobe,0,0,0)[1] < f._B(myExBprobe,0,0,z_check)[0]/f._B(myExBprobe,0,0,0)[0]:
    print('B > E: ions deflected -y (require lower v_w)')
    search_right_fist = True
else: 
    print('B < E: ions deflected +y (require higher v_w)')
    search_right_fist = False

start = datetime.now()
BC_result_show_TF = False # Ion's final position result show True/False
for n in range(0,len(v_ion_array)):
    v_x_0    = np.ones(N_points)*v_ion_array[n]*np.cos(α_y*np.pi/180)*np.sin(α_x*np.pi/180) # m/s
    v_y_0    = np.ones(N_points)*v_ion_array[n]*np.sin(α_y*np.pi/180) # m/s
    v_z_0    = np.ones(N_points)*v_ion_array[n]*np.cos(α_y*np.pi/180)*np.cos(α_x*np.pi/180) # m/s
    r_0      = np.array((x_0,y_0,z_0,v_x_0,v_y_0,v_z_0))
    t_max    = abs((myExBprobe.l_c+myExBprobe.l_f+myExBprobe.l_d)/v_ion_array[n])*2 # maximum time limit, s
    t_gyro   = 2*np.pi*m_ion/(q_ion*myExBprobe.Bxpra) # gyro period, s
    # print('dt_max = {:.2e} s, dt_gyro = {:.2e} s'.format(t_max/100,t_gyro/25))
    dt       = np.min([t_max/100, t_gyro/25]) # time step, s
    t_array  = np.arange(0,t_max,dt) # time array, s

    m = n
    count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)
    print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))

    if search_right_fist == True:
        while count_collected[m,n] == 0 and m > 0:
            m -= 1
            print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
            count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)
        while count_lost[m,n] != N_points and m > 0:
            m -= 1
            print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
            count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)
        m = n
        while count_lost[m,n] != N_points and m < M-1:
            m += 1
            print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
            count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)

    else:
        # while count_collected[m,n] == 0 and m < M-1:
        #     m += 1
        #     print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
        #     count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array)
        while count_lost[m,n] != N_points and m < M-1:
            m += 1
            print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
            count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)
        m = n
        while count_lost[m,n] != N_points and m > 0:
            m -= 1
            print('(n,m) = ({},{}) out of ({},{}) & (v_ion,v_wpra) = ({:.1f}, {:.1f}) km/s'.format(n+1,m+1,N,M,v_ion_array[n]/1e3,vwpra_array[m]/1e3))
            count_lost[m,n], count_collected[m,n] = p._transmittancy_calculation(myExBprobe,ion,vwpra_array[m],r_0,t_array,BC_result_show_TF)
    
end = datetime.now()
total_time_seconds = (end - start).total_seconds()
print('Total computation time : {} s'.format(total_time_seconds))
print('Average computation time: {} s'.format(total_time_seconds/N_points/len(v_ion_array)/len(vwpra_array)))

#% Numerical Transmittancy Matrix Export
FS = 8 # font size in poin
MS = 1 # marker size
LW = 1 # line width
DPI = 1200 # dot per inch
figure_size = (3.37,3.37)

XX, YY = np.meshgrid(vwpra_array/1e3, v_ion_array/1e3)
fig, axs = plt.subplots(1,1,figsize=(160/25.4,80/25.4),dpi=DPI,facecolor='w',sharex=True,sharey=True)
CS = axs.pcolormesh(XX,YY, count_collected.transpose()/N_points, shading='auto',cmap='viridis',norm=mcolors.Normalize(vmin=0,vmax=1))
for i in range(M-1): axs.plot((XX[0,i]+XX[0,i+1])/2* np.ones(N),YY[:,0],c='k',ls='-',linewidth=0.1)
for j in range(N-1): axs.plot( XX[0,:]           ,(YY[j,:]+YY[j+1,:])/2,c='k',ls='-',linewidth=0.1)
axs.plot(vwpra_array/1e3, (vwpra_array+Δv_w_array)/1e3, 'r--', lw=LW/2)
axs.plot(vwpra_array/1e3, (vwpra_array-Δv_w_array)/1e3, 'r--', lw=LW/2)
axs.set_title('{} ({:.0f} sec)'.format(ion_name,total_time_seconds) ,fontsize=FS)
axs.set_ylabel('$v_{ion}$, km/s',fontsize=FS)
axs.set_xlabel('$v_w$, km/s',fontsize=FS)
axs.set_xlim((0,123))
axs.set_ylim(axs.get_xlim())
axs.set_xticks(np.arange(axs.get_xlim()[0],axs.get_xlim()[1]+1,20))
axs.set_yticks(axs.get_xticks())
axs.tick_params(axis='both', which='major', labelsize=FS)
axs.tick_params(axis='both', which='minor', labelsize=FS)
axs.set_aspect('equal', adjustable='box')
clb = plt.colorbar(CS, ax=axs, ticks=np.linspace(0, 1.0, 11),shrink=1, pad=0.01)
clb.ax.tick_params(labelsize=FS)
clb.set_label(label='Transmittancy level',size=FS)

T_k = np.zeros((M+1,N+1),dtype=object)
T_k[0,1:] = v_ion_array
T_k[1:,0] = vwpra_array
T_k[1:,1:] = count_collected/N_points
T_k[0,0] = 'v_w[m/s]\\v_ion[m/s];alpha_x={:.1f}[deg];alpha_y={:.1f}[deg];breast stroke;{}s;{}'.format(α_x,α_y,total_time_seconds,datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
np.savetxt( myExBprobe.directory+'T_{}_n (M,N,N_points) = ({},{},{}), v = [{}, {}] kms {} (α_x,α_y) = ({:.2f}°,{:.2f}°).txt'.format(ion_name,M,N,N_points,np.min(v_ion_array)/1e3,np.max(v_ion_array)/1e3,myExBprobe.memo,α_x,α_y),T_k,fmt='%s',delimiter=',')
plt.savefig(myExBprobe.directory+'T_{}_n (M,N,N_points) = ({},{},{}), v = [{}, {}] kms {} (α_x,α_y) = ({:.2f}°,{:.2f}°).png'.format(ion_name,M,N,N_points,np.min(v_ion_array)/1e3,np.max(v_ion_array)/1e3,myExBprobe.memo,α_x,α_y),dpi=DPI, bbox_inches='tight')