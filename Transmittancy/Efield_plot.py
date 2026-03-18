#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import _ExB_probe_info as _ExB_probe_info
myExBprobe = _ExB_probe_info.Sample_ExB_probe
print('Chosen ExB probe: {}'.format(myExBprobe))

import _transmittancy_calc as _transmittancy_calc
f = _transmittancy_calc.field()

z_array = np.linspace(-myExBprobe.l_f/2 - myExBprobe.l_c,
                       myExBprobe.l_f/2 + myExBprobe.l_d,500)

E_name_backup = myExBprobe.filename_E
Ex_array = np.zeros(len(z_array))
Ey_array = np.zeros(len(z_array))
Ez_array = np.zeros(len(z_array))
Po_array = np.zeros(len(z_array))
Ey0 = f._E(myExBprobe,0,0,0)[1]
for i in range(len(z_array)):
    Ex_array[i] = f._E(myExBprobe,0,0,z_array[i])[0]
    Ey_array[i] = f._E(myExBprobe,0,0,z_array[i])[1]
    Ez_array[i] = f._E(myExBprobe,0,0,z_array[i])[2]
    Po_array[i] = f._E(myExBprobe,0,0,z_array[i])[3]

myExBprobe.filename_E = "Sample_ExB_probe_E_2V.txt"
Ex_array_2 = np.zeros(len(z_array))
Ey_array_2 = np.zeros(len(z_array))
Ez_array_2 = np.zeros(len(z_array))
P_array_2  = np.zeros(len(z_array))
Ey0_2 = f._E(myExBprobe,0,0,0)[1]
P0_2  = f._E(myExBprobe,0,0,0)[3]
for i in range(len(z_array)):
    Ex_array_2[i] = f._E(myExBprobe,0,0,z_array[i])[0]
    Ey_array_2[i] = f._E(myExBprobe,0,0,z_array[i])[1]
    Ez_array_2[i] = f._E(myExBprobe,0,0,z_array[i])[2]
    P_array_2 [i] = f._E(myExBprobe,0,0,z_array[i])[3]

myExBprobe.filename_E = "Sample_ExB_probe_E_40V.txt"
Ex_array_3 = np.zeros(len(z_array))
Ey_array_3 = np.zeros(len(z_array))
Ez_array_3 = np.zeros(len(z_array))
P_array_3  = np.zeros(len(z_array))
Ey0_3 = f._E(myExBprobe,0,0,0)[1]
P0_3  = f._E(myExBprobe,0,0,0)[3]
for i in range(len(z_array)):
    Ex_array_3[i] = f._E(myExBprobe,0,0,z_array[i])[0]
    Ey_array_3[i] = f._E(myExBprobe,0,0,z_array[i])[1]
    Ez_array_3[i] = f._E(myExBprobe,0,0,z_array[i])[2]
    P_array_3 [i] = f._E(myExBprobe,0,0,z_array[i])[3]

#%% Plot
V_electrodes = np.array([10,2,40]) # V
Color = np.array(('k','r','b'))

FS = 8 # font size in point
MS = 2 # marker size
LW = 1 # line width 
DPI = 300 # dot per inch

figure_size = (3.25,2.00)
fig, ax = plt.subplots(1,1,figsize=figure_size,constrained_layout=True,facecolor='w')
ax.plot(z_array/myExBprobe.l_f,Ey_array  *myExBprobe.d_e/V_electrodes[0],ls='-' ,lw=4*LW,c=Color[0],label=V_electrodes[0])
ax.plot(z_array/myExBprobe.l_f,Ey_array_2*myExBprobe.d_e/V_electrodes[1],ls='--',lw=3*LW,c=Color[1],label=V_electrodes[1])
ax.plot(z_array/myExBprobe.l_f,Ey_array_3*myExBprobe.d_e/V_electrodes[2],ls=':' ,lw=2*LW,c=Color[2],label=V_electrodes[2])

ax.set_title('E-field normalized at $z=0$',fontsize=FS)
ax.set_xlabel('$z / l_f$',fontsize=FS)
ax.set_ylabel('$\\frac{E_y(z/l_f)\\cdot d_e}{V_{electrodes}}$',fontsize=FS)

ax.set_xlim((-0.75,0.75))
ax.set_xticks(np.linspace(*ax.get_xlim(),7))
ax.set_ylim((-0.1,1.1))
ax.set_yticks(np.linspace(0.0,1.0,5))

ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c )/myExBprobe.l_f,ax.get_xlim()[0]]                  ,*ax.get_ylim(),alpha=0.2,color='k'         )
ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c )/myExBprobe.l_f,-(myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:blue'  )
ax.fill_between([-(myExBprobe.l_f/2                )/myExBprobe.l_f, (myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:orange')
ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f,ax.get_xlim()[1]]                  ,*ax.get_ylim(),alpha=0.2,color='k'         )
ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f, (myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:green' )

ax.xaxis.offsetText.set_fontsize(FS)
ax.yaxis.offsetText.set_fontsize(FS)
ax.tick_params(axis='both',labelsize=FS)
ax.grid()
ax.legend(loc='best',framealpha=1.0,fontsize=FS,title='$V_{electrodes}$, V',title_fontsize=FS,handlelength=3.5)

plt.tight_layout()
plt.savefig(myExBprobe.directory+'Efield_'+myExBprobe.memo+'.png',dpi=DPI)
plt.show()

#% x-, y-, z-component of E-field
figure_size = (6.5,5)

fig, axs = plt.subplots(3,2,figsize=figure_size,facecolor='w',linewidth=0,edgecolor='k',constrained_layout=True)
axs[0,0].plot(z_array/myExBprobe.l_f,Ex_array  *myExBprobe.d_e,ls='-' ,lw=4*LW,c=Color[0])
axs[0,0].plot(z_array/myExBprobe.l_f,Ex_array_2*myExBprobe.d_e,ls='--',lw=3*LW,c=Color[1])
axs[0,0].plot(z_array/myExBprobe.l_f,Ex_array_3*myExBprobe.d_e,ls=':' ,lw=2*LW,c=Color[2])
axs[1,0].plot(z_array/myExBprobe.l_f,Ey_array  *myExBprobe.d_e,ls='-' ,lw=4*LW,c=Color[0])
axs[1,0].plot(z_array/myExBprobe.l_f,Ey_array_2*myExBprobe.d_e,ls='--',lw=3*LW,c=Color[1])
axs[1,0].plot(z_array/myExBprobe.l_f,Ey_array_3*myExBprobe.d_e,ls=':' ,lw=2*LW,c=Color[2])
axs[2,0].plot(z_array/myExBprobe.l_f,Ez_array  *myExBprobe.d_e,ls='-' ,lw=4*LW,c=Color[0])
axs[2,0].plot(z_array/myExBprobe.l_f,Ez_array_2*myExBprobe.d_e,ls='--',lw=3*LW,c=Color[1])
axs[2,0].plot(z_array/myExBprobe.l_f,Ez_array_3*myExBprobe.d_e,ls=':' ,lw=2*LW,c=Color[2])

axs[0,1].plot(z_array/myExBprobe.l_f,Ex_array  /np.max(Ey_array  ),ls='-' ,lw=4*LW,c=Color[0])
axs[0,1].plot(z_array/myExBprobe.l_f,Ex_array_2/np.max(Ey_array_2),ls='--',lw=3*LW,c=Color[1])
axs[0,1].plot(z_array/myExBprobe.l_f,Ex_array_3/np.max(Ey_array_3),ls=':' ,lw=2*LW,c=Color[2])
axs[1,1].plot(z_array/myExBprobe.l_f,Ey_array  /np.max(Ey_array  ),ls='-' ,lw=4*LW,c=Color[0])
axs[1,1].plot(z_array/myExBprobe.l_f,Ey_array_2/np.max(Ey_array_2),ls='--',lw=3*LW,c=Color[1])
axs[1,1].plot(z_array/myExBprobe.l_f,Ey_array_3/np.max(Ey_array_3),ls=':' ,lw=2*LW,c=Color[2])
axs[2,1].plot(z_array/myExBprobe.l_f,Ez_array  /np.max(Ey_array  ),ls='-' ,lw=4*LW,c=Color[0])
axs[2,1].plot(z_array/myExBprobe.l_f,Ez_array_2/np.max(Ey_array_2),ls='--',lw=3*LW,c=Color[1])
axs[2,1].plot(z_array/myExBprobe.l_f,Ez_array_3/np.max(Ey_array_3),ls=':' ,lw=2*LW,c=Color[2])

axs[2,0].set_title('E-field',fontsize=FS)
axs[2,1].set_title('Normalized E-field',fontsize=FS)

axs[2,0].set_xlabel('$z$, mm',fontsize=FS)
axs[2,1].set_xlabel('$z$, mm',fontsize=FS)

axs[0,0].set_ylabel('$E_x \\cdot d_e$, V',fontsize=FS)
axs[1,0].set_ylabel('$E_y \\cdot d_e$, V',fontsize=FS)
axs[2,0].set_ylabel('$E_z \\cdot d_e$, V',fontsize=FS)
axs[0,1].set_ylabel('Normalized $E_x$, -',fontsize=FS)
axs[1,1].set_ylabel('Normalized $E_y$, -',fontsize=FS)
axs[2,1].set_ylabel('Normalized $E_z$, -',fontsize=FS)

axs[0,0].set_ylim((-0.6,0.6))
axs[1,0].set_ylim((   0,42))
axs[2,0].set_ylim((-0.6,0.6))
axs[0,1].set_ylim((-0.02,0.02))
axs[1,1].set_ylim((-0.100,1.100))
axs[2,1].set_ylim((-0.02,0.02))

abcdef = np.array(['(a)','(d)','(b)','(e)','(c)','(f)'])
i = 0
for ax in axs.flat:
    ax.text(0.05,0.90,abcdef[i],transform=ax.transAxes,fontsize=FS,va='top',ha='left',bbox=dict(facecolor='w', edgecolor='none', alpha=1.0))
    ax.set_xlim((-1.1,1.1))
    ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c )/myExBprobe.l_f,ax.get_xlim()[0]]                  ,*ax.get_ylim(),alpha=0.2,color='k'         )
    ax.fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c )/myExBprobe.l_f,-(myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:blue'  )
    ax.fill_between([-(myExBprobe.l_f/2                )/myExBprobe.l_f, (myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:orange')
    ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f,ax.get_xlim()[1]]                  ,*ax.get_ylim(),alpha=0.2,color='k'         )
    ax.fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f, (myExBprobe.l_f/2)/myExBprobe.l_f],*ax.get_ylim(),alpha=0.2,color='tab:green' )

    i += 1

    ax.xaxis.offsetText.set_fontsize(FS)
    ax.yaxis.offsetText.set_fontsize(FS)
    ax.tick_params(axis='both',labelsize=FS)
    # ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    ax.grid()

axs[1,1].legend(V_electrodes,loc='best',framealpha=1.0,fontsize=FS,title='$V_{electrodes}$, V',title_fontsize=FS,handlelength=3.5)

plt.savefig(myExBprobe.directory+'E-fields vs z with different V_electrodes '+myExBprobe.memo+'.png',dpi=DPI,bbox_inches='tight',facecolor=fig.get_facecolor(),edgecolor=fig.get_edgecolor())
plt.show()