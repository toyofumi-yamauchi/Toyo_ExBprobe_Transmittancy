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

#%%
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
n_last_array    = np.zeros(N_points)
F_n_array       = np.zeros((N_points,len(t_array),7))
KE_array        = np.zeros((N_points,len(t_array)))
PE_array        = np.zeros((N_points,len(t_array)))
TE_array        = np.zeros((N_points,len(t_array)))
z_array         = np.linspace(-myExBprobe.l_f/2 - myExBprobe.l_c, myExBprobe.l_f/2 + myExBprobe.l_d,1000)
Ey_plot         = np.zeros((N_points,len(z_array)))
Bx_plot         = np.zeros((N_points,len(z_array)))
Fy_plot         = np.zeros((N_points,len(z_array)))
fy_plot         = np.zeros((N_points,len(t_array)))

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

    TE_ini = 0.5*m_ion/q_ion*v_ion**2 + F_n_array[n,0,6] # eV
    KE_array[n,:int(n_last_array[n])] = 0.5*m_ion/q_ion*(r_n_array[n,:int(n_last_array[n]),3]**2 + r_n_array[n,:int(n_last_array[n]),4]**2 + r_n_array[n,:int(n_last_array[n]),5]**2) # eV
    PE_array[n,:int(n_last_array[n])] = F_n_array[n,:int(n_last_array[n]),6]
    TE_array[n,:int(n_last_array[n])] = KE_array[n,:int(n_last_array[n])] + PE_array[n,:int(n_last_array[n])]

    for l in range(0,len(z_array)):
       Ey_plot[n,l] = f._E(myExBprobe,x_0[n],y_0[n],z_array[l],vwpra)[1]
       Bx_plot[n,l] = f._B(myExBprobe,x_0[n],y_0[n],z_array[l])[0]
    Fy_plot[n,:] = q_ion*(v_z_0[n]*Bx_plot[n,:]+Ey_plot[n,:])
    fy_plot[n,:int(n_last_array[n])] = q_ion*(r_n_array[n,:int(n_last_array[n]),5]*F_n_array[n,:int(n_last_array[n]),0]+F_n_array[n,:int(n_last_array[n]),4])

TE_array[TE_array == 0] = np.nan

end = datetime.now()
print('Total computation time : {} s'.format((end - start).total_seconds()))
print('Average computation time: {} s'.format((end - start).total_seconds()/N_points))
print('Lost     : {} / {}'.format(count_lost ,N_points))
print('Collected: {} / {}'.format(count_collected,N_points))

#%% Plot
z_check = -myExBprobe.l_f*0.4
if f._E(myExBprobe,0,0,z_check)[1]/f._E(myExBprobe,0,0,0)[1] < f._B(myExBprobe,0,0,z_check)[0]/f._B(myExBprobe,0,0,0)[0]:
    print('B > E: ions deflected -y (require lower v_w)')
else:
    print('B < E: ions deflected +y (require higher v_w)')

FS = 8 # font size in point
MS = 1 # marker size
LW = 1 # line width 
DPI = 300 # dot per inch

figure_size = (3.25,8)
fig, axs = plt.subplots(5,1,figsize=figure_size,constrained_layout=True,facecolor='w')
for n in range(0,N_points):
    lnE = axs[0].plot(z_array*1e3,                              Ey_plot[n,:]                    /np.max(Ey_plot[n,:]),                          ls='-' , lw=LW,c='tab:red',   label='$E_y(z)/E_{y,0}$')
    lnB = axs[0].plot(z_array*1e3,                              Bx_plot[n,:]                    /np.min(Bx_plot[n,:]),                          ls='--', lw=LW,c='tab:purple',label='$B_x(z)/B_{x,0}$')
    axs[0].plot(z_check*1e3,f._E(myExBprobe,0,0,z_check)[1]/f._E(myExBprobe,0,0,0)[1],ls='none',lw=LW,c='tab:red'   ,marker='x',ms=8)
    axs[0].plot(z_check*1e3,f._B(myExBprobe,0,0,z_check)[0]/f._B(myExBprobe,0,0,0)[0],ls='none',lw=LW,c='tab:purple',marker='x',ms=8)
    axs
    axs[1].plot(r_n_array[n,                    0,2]*1e3,r_n_array[n,                 0,0]*1e3,ls='none',lw=LW,c='b',marker='.',ms=5)
    axs[1].plot(r_n_array[n, int(n_last_array[n]),2]*1e3,r_n_array[n, int(n_last_array[n]),0]*1e3,ls='none',lw=LW,c='r',marker='.',ms=5)
    axs[1].plot(r_n_array[n,:int(n_last_array[n]),2]*1e3,r_n_array[n,:int(n_last_array[n]),0]*1e3,ls='none',lw=LW,c='k',marker='.',ms=MS)

    axs[2].plot(r_n_array[n,               0,2]*1e3,r_n_array[n,               0,1]*1e3,ls='none',lw=LW,c='b',marker='.',ms=5)
    axs[2].plot(r_n_array[n, int(n_last_array[n]),2]*1e3,r_n_array[n, int(n_last_array[n]),1]*1e3,ls='none',lw=LW,c='r',marker='.',ms=5)
    axs[2].plot(r_n_array[n,:int(n_last_array[n]),2]*1e3,r_n_array[n,:int(n_last_array[n]),1]*1e3,ls='none',lw=LW,c='k',marker='.',ms=MS)

    axs[3].plot(t_array[:int(n_last_array[n])],(TE_array[n,:int(n_last_array[n])]/TE_array[0,:int(n_last_array[n])]-1)*100,ls='-', lw=LW,c='k',label='Energy change ($E(t=0) = {:.0f} eV$)'.format(TE_ini))

    axs[4].plot(t_array[:int(n_last_array[n])],(r_n_array[n,:int(n_last_array[n]),5]/v_ion-1)*100 ,ls='-', lw=LW,c='k',label='Z-velocity change ($v_z$(t=0) = {:.1f} km/s'.format(v_ion/1000))
    
axs[0].set_title('E- and B-field normalized along z-axis',fontsize=FS)
axs[1].set_title('Ion Trajectory in the xz plane',fontsize=FS)
axs[2].set_title('Ion Trajectory in the yz plane',fontsize=FS)
axs[3].set_title('Energy Conservation: $Total(t) = KE(t) + PE(t)$',fontsize=FS)
axs[4].set_title('Z-velocity check: $v_z(t)$',fontsize=FS)

axs[0].set_ylabel('Normalized field, -',fontsize=FS)
axs[1].set_ylabel('$x$, mm',fontsize=FS)
axs[2].set_ylabel('$y$, mm',fontsize=FS)
axs[3].set_ylabel('$TE(t)/TE(t=0)$-1, %',fontsize=FS)
axs[4].set_ylabel('$v_z(t)/v_z(t=0)$-1, %',fontsize=FS)

axs[0].set_ylim((-1.1,1.1))
axs[1].set_ylim((-myExBprobe.d_m*1e3,myExBprobe.d_m*1e3))
axs[2].set_ylim((-myExBprobe.d_e*1e3,myExBprobe.d_e*1e3)) # mm
axs[1].set_ylim((-30,30))
axs[2].set_ylim((-30,30)) # m

for ax in np.array([axs[0],axs[1],axs[2]]):
    ax.set_xlabel('$z$, mm',fontsize=FS)
    ax.set_xlim((-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,(myExBprobe.l_f/2+myExBprobe.l_d+0.05)*1e3)) # mm
    ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,ax.get_xlim()[0]]       ,*ax.get_ylim(),alpha=0.2,color='k'         )
    ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d)*1e3,ax.get_xlim()[1]]       ,*ax.get_ylim(),alpha=0.2,color='k'         )
    ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,-(myExBprobe.l_f/2)*1e3],*ax.get_ylim(),alpha=0.2,color='tab:blue'  )
    ax.fill_between([-(myExBprobe.l_f/2               )*1e3, (myExBprobe.l_f/2)*1e3],*ax.get_ylim(),alpha=0.2,color='tab:orange')
    ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d)*1e3, (myExBprobe.l_f/2)*1e3],*ax.get_ylim(),alpha=0.2,color='tab:green' )

axs[1].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,-(myExBprobe.l_f/2)*1e3], myExBprobe.r_t  *1e3, axs[1].get_ylim()[1],alpha=0.8,color='k') # collimator left
axs[1].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,-(myExBprobe.l_f/2)*1e3],-myExBprobe.r_t  *1e3,-axs[1].get_ylim()[1],alpha=0.8,color='k') # collimator right
axs[1].fill_between([-(myExBprobe.l_m/2               )*1e3, (myExBprobe.l_m/2)*1e3], myExBprobe.d_m/2*1e3, axs[1].get_ylim()[1],alpha=0.8,color='tab:purple') # magnet left
axs[1].fill_between([-(myExBprobe.l_m/2               )*1e3, (myExBprobe.l_m/2)*1e3],-myExBprobe.d_m/2*1e3,-axs[1].get_ylim()[1],alpha=0.8,color='tab:purple') # magnet right
axs[1].fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d)*1e3, (myExBprobe.l_f/2)*1e3], myExBprobe.r_t  *1e3, axs[1].get_ylim()[1],alpha=0.8,color='k') # drift tube left
axs[1].fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d)*1e3, (myExBprobe.l_f/2)*1e3],-myExBprobe.r_t  *1e3,-axs[1].get_ylim()[1],alpha=0.8,color='k') # drift tube right
axs[2].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,-(myExBprobe.l_f/2)*1e3], myExBprobe.r_t  *1e3, axs[2].get_ylim()[1],alpha=0.8,color='k') # collimator top
axs[2].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,-(myExBprobe.l_f/2)*1e3],-myExBprobe.r_t  *1e3,-axs[2].get_ylim()[1],alpha=0.8,color='k') # collimator bottom

axs[1].fill_between([ (myExBprobe.l_f/2               )*1e3,axs[1].get_xlim()[1]], myExBprobe.r_t*1e3, axs[1].get_ylim()[1],alpha=0.8,color='k') # drift tube left
axs[1].fill_between([ (myExBprobe.l_f/2               )*1e3,axs[1].get_xlim()[1]],-myExBprobe.r_t*1e3,-axs[1].get_ylim()[1],alpha=0.8,color='k') # drift tube right
axs[2].fill_between([ (myExBprobe.l_f/2               )*1e3,axs[2].get_xlim()[1]], myExBprobe.r_t*1e3, axs[2].get_ylim()[1],alpha=0.8,color='k') # drift tube top
axs[2].fill_between([ (myExBprobe.l_f/2               )*1e3,axs[2].get_xlim()[1]],-myExBprobe.r_t*1e3,-axs[2].get_ylim()[1],alpha=0.8,color='k') # drift tube bottom
axs[1].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,axs[1].get_xlim()[0]], myExBprobe.r_1*1e3, axs[1].get_ylim()[1],alpha=0.8,color='k') # collector
axs[1].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,axs[1].get_xlim()[0]],-myExBprobe.r_1*1e3,-axs[1].get_ylim()[1],alpha=0.8,color='k') # collector
axs[2].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,axs[2].get_xlim()[0]], myExBprobe.r_1*1e3, axs[2].get_ylim()[1],alpha=0.8,color='k') # collector
axs[2].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c)*1e3,axs[2].get_xlim()[0]],-myExBprobe.r_1*1e3,-axs[2].get_ylim()[1],alpha=0.8,color='k') # collector
axs[1].fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d)*1e3,axs[1].get_xlim()[1]],-myExBprobe.r_4*1e3, myExBprobe.r_4*1e3  ,alpha=0.8,color='gold') # collector

axs[2].fill_between(myExBprobe.z_electrodes*1e3, myExBprobe.y_electrodes*1e3,alpha=0.8,color='tab:red',lw=0)
axs[2].fill_between(myExBprobe.z_electrodes*1e3,-myExBprobe.y_electrodes*1e3,alpha=0.8,color='tab:red',lw=0)

t_range_min = 0
t_range_max = t_max/2
for ax in np.array([axs[3],axs[4]]):
    ax.set_xlabel('$t$, sec',fontsize=FS)
    ax.set_xlim((t_range_min,t_range_max))

for ax in axs.flat:
    ax.xaxis.offsetText.set_fontsize(FS)
    ax.yaxis.offsetText.set_fontsize(FS)
    ax.tick_params(axis='both',labelsize=FS)

# plt.tight_layout()
plt.savefig(myExBprobe.directory+'Ion Trajectory ({}, v_ion = {:.1f} kms^-1, v_w = {:.1f} kms^-1).png'.format(myExBprobe.memo, v_ion/1e3, vwpra/1e3),dpi=DPI)
plt.show()

