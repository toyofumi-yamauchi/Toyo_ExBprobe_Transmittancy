#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os

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

#%%
T = np.zeros((M,N))
start = datetime.now()
for n in range(0,len(v_ion_array)):
    for m in range(0,len(vwpra_array)):
        T[m,n] = p._transmittancy_calculation_analitycal(myExBprobe,ion,vwpra_array[m],v_ion_array[n],α_x,α_y)
end = datetime.now()
print('Total computation time : {} s'.format((end - start).total_seconds()))
print('Average computation time: {} s'.format((end - start).total_seconds()/M/N))

#% Analytical Transmittancy Matrix Export
FS = 8 # font size in point
MS = 1 # marker size
LW = 1 # line width
DPI = 300 # dot per inch
figure_size = (3.37,3.37)

XX, YY = np.meshgrid(vwpra_array/1e3, v_ion_array/1e3)
fig, axs = plt.subplots(1,1,figsize=(160/25.4,80/25.4),dpi=DPI,facecolor='w',sharex=True,sharey=True)
CS = axs.pcolormesh(XX,YY, T.transpose(), shading='auto',cmap='viridis',norm=mcolors.Normalize(vmin=0,vmax=1))
for i in range(M-1): axs.plot((XX[0,i]+XX[0,i+1])/2* np.ones(N),YY[:,0],c='k',ls='-',linewidth=0.1)
for j in range(N-1): axs.plot( XX[0,:]           ,(YY[j,:]+YY[j+1,:])/2,c='k',ls='-',linewidth=0.1)
axs.plot(vwpra_array/1e3, (vwpra_array+Δv_w_array)/1e3, 'r--', lw=LW/2)
axs.plot(vwpra_array/1e3, (vwpra_array-Δv_w_array)/1e3, 'r--', lw=LW/2)
axs.set_title('{}'.format(ion_name) ,fontsize=FS)
axs.set_ylabel('$v_{ion}$, km/s',fontsize=FS)
axs.set_xlabel('$v_w$, km/s',fontsize=FS)
axs.set_xlim((0,v_max/1e3))
axs.set_ylim(axs.get_xlim())
axs.set_xticks(np.arange(0, v_max/1e3+20, 20))
axs.set_yticks(axs.get_xticks())
axs.tick_params(axis='both', which='major', labelsize=FS)
axs.tick_params(axis='both', which='minor', labelsize=FS)
axs.set_aspect('equal', adjustable='box')
clb = plt.colorbar(CS, ax=axs, ticks=np.linspace(0, 1.0, 11),shrink=1, pad=0.02)
clb.ax.tick_params(labelsize=FS)
clb.set_label(label='Transmittancy level',size=FS)

T_k = np.zeros((M+1,N+1),dtype=object)
T_k[0,1:] = v_ion_array
T_k[1:,0] = vwpra_array
T_k[1:,1:] = T
T_k[0,0] = 'v_w[m/s]\\v_ion[m/s];alpha_x={:.1f}[deg];alpha_y={:.1f}[deg];Analytical;{}'.format(α_x,α_y,datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
np.savetxt( myExBprobe.directory+'T_{}_a (M,N) = ({},{}), v = [{}, {}] kms {} (α_x,α_y) = ({:.2f}°,{:.2f}°).txt'.format(ion_name,M,N,np.min(v_ion_array)/1e3,np.max(v_ion_array)/1e3,myExBprobe.memo,α_x,α_y),T_k,fmt='%s',delimiter=',')
plt.savefig(myExBprobe.directory+'T_{}_a (M,N) = ({},{}), v = [{}, {}] kms {} (α_x,α_y) = ({:.2f}°,{:.2f}°).png'.format(ion_name,M,N,np.min(v_ion_array)/1e3,np.max(v_ion_array)/1e3,myExBprobe.memo,α_x,α_y),dpi=DPI, bbox_inches='tight')
