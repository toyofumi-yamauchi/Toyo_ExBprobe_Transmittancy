#%%
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import _ExB_probe_info as _ExB_probe_info
myExBprobe = _ExB_probe_info.Sample_ExB_probe
print('Chosen ExB probe: {}'.format(myExBprobe))

from Transmittancy import _transmittancy_calc as _transmittancy_calc
f = _transmittancy_calc.field()

from _transmittancy_calc import _sunflower_seed
N_points = 10
r_ion    = 0.99*myExBprobe.r_1
x0, y0   = _sunflower_seed(N_points,r_ion)
r0       = np.sqrt(x0**2+y0**2)

z_array = np.linspace(-myExBprobe.l_f/2 - myExBprobe.l_c,
                       myExBprobe.l_f/2 + myExBprobe.l_d,500)
dl = z_array[1] - z_array[0]

Ey0 = f._E(myExBprobe,0,0,0)[1]
Bx0 = f._B(myExBprobe,0,0,0)[0]
Bx_array = np.zeros((len(x0),len(z_array)))
Ey_array = np.zeros((len(x0),len(z_array)))
B_praness = np.zeros((len(x0)))
R_squared = np.zeros((len(x0)))
for i in range(0,len(x0)):
    Bxe = 0
    Eye = 0
    for j in range(0,len(z_array)):
        Bx_array[i,j] = f._B(myExBprobe,x0[i],y0[i],z_array[j])[0]
        Ey_array[i,j] = f._E(myExBprobe,x0[i],y0[i],z_array[j])[1]

        Bxe += dl/myExBprobe.l_f*Bx_array[i,j] # Tesla
        Eye += dl/myExBprobe.l_f*Ey_array[i,j] # V/m
    B_praness[i] = (Bxe/Bx0)/(Eye/Ey0)
    R_squared[i] = np.corrcoef(Ey_array[i,:]/np.max(Ey_array[i,:]),Bx_array[i,:]/np.max(Bx_array[i,:]))[0,1]**2

    print('Practicalness = {:.3%} (B = {:.3%}, E = {:.3%})'.format(B_praness[i],Bxe/Bx0,Eye/Ey0))
    print('R^2: {}'.format(R_squared[i]))
print('-------------------------------')
print('Mean Practicalness = {:.3%}±{:.3%}'.format(np.mean(B_praness),np.std(B_praness)))
print('Mean R^2 = {:.5f}±{:.5f}'.format(np.mean(R_squared),np.std(R_squared)))

#%% Plot
FS = 8 # font size in point
MS = 5 # marker size
LW = 2 # line width 
DPI = 300 # dot per inch

figure_size = (3.25,2.00)
fig, ax = plt.subplots(1,1,figsize=figure_size,constrained_layout=True,facecolor='w')
ax.plot(z_array/myExBprobe.l_f,Ey_array[0,:]/Ey0,ls='-' ,lw=LW,c='tab:red',   label='E-field')
ax.plot(z_array/myExBprobe.l_f,Bx_array[0,:]/Bx0,ls='--',lw=LW,c='tab:purple',label='B-field')

ax.set_title('E- and B-field normalized at $z=0$',fontsize=FS)
ax.set_xlabel('$z / l_f$',fontsize=FS)
ax.set_ylabel('$\\frac{E_y(z/l_f)}{E_{y,0}}$ or $\\frac{B_x(z/l_f)}{B_{x,0}}$',fontsize=FS)
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
ax.legend(loc='best',framealpha=1.0,fontsize=FS)

plt.tight_layout()
plt.savefig(myExBprobe.directory+'ExBfield.png',dpi=DPI)
plt.show()

#%
figure_size = (6.5,2.15)
Position_Color = plt.cm.bwr(np.linspace(0,1,len(x0)))
Position_Color = plt.cm.gist_gray(np.linspace(0,1,len(r0)))
Position_Color = plt.cm.jet(np.linspace(0,1,len(r0)))

fig, axs = plt.subplots(1,3,figsize=figure_size,facecolor='w',linewidth=0,edgecolor='k')
for i in range(0,len(x0)):
    axs[0].plot([1e3*x0[i],1e3*x0[i]],[1e3*y0[i],1e3*y0[i]],ls='none',c=Position_Color[i],marker='.',ms=MS)
    axs[1].plot(z_array/myExBprobe.l_f,Bx_array[i,:]/Bx0,ls='-',lw=5-4*i/len(x0),c=Position_Color[i])
    axs[2].plot(z_array/myExBprobe.l_f,Ey_array[i,:]/Ey0,ls='-',lw=5-4*i/len(x0),c=Position_Color[i])

axs[0].add_patch(plt.Circle((0,0),1e3*myExBprobe.r_1,color='k',fill=False))
axs[0].set_aspect('equal')
axs[0].set_xlabel('$x$, mm',fontsize=FS)
axs[0].set_ylabel('$y$, mm',fontsize=FS)
axs[0].set_title('(x,y) points\n(Looking from the back\n of the ExB probe)',fontsize=FS)

axs[1].set_ylabel('$\\frac{B_x(z/l_f)}{B_{x,0}}$',fontsize=FS)
axs[2].set_ylabel('$\\frac{E_y(z/l_f)}{E_{y,0}}$',fontsize=FS)
axs[1].set_title('B-field',fontsize=FS)
axs[2].set_title('E-field',fontsize=FS)

for i in range(1,3):
    axs[i].set_xlabel('$z / l_f$',fontsize=FS)
    axs[i].set_xlim((-1.1,1.1))
    axs[i].set_ylim((-0.1,1.1))
    axs[i].fill_between([-(myExBprobe.l_f/2+myExBprobe.l_c )/myExBprobe.l_f,axs[i].get_xlim()[0]]                ,*axs[i].get_ylim(),alpha=0.2,color='k'         )
    axs[i].fill_betweenx(axs[i].get_ylim(),( -myExBprobe.l_c-myExBprobe.l_f/2)/myExBprobe.l_f,(    -myExBprobe.l_f/2)/myExBprobe.l_f,alpha=0.2,color='tab:blue')
    axs[i].fill_betweenx(axs[i].get_ylim(),(                -myExBprobe.l_f/2)/myExBprobe.l_f,(     myExBprobe.l_f/2)/myExBprobe.l_f,alpha=0.2,color='tab:orange')
    axs[i].fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f, axs[i].get_xlim()[1]               ],*axs[i].get_ylim(),alpha=0.2,color='k'         )
    axs[i].fill_between([ (myExBprobe.l_f/2+myExBprobe.l_d )/myExBprobe.l_f, (myExBprobe.l_f/2)  /myExBprobe.l_f],*axs[i].get_ylim(),alpha=0.2,color='tab:green' )

for ax in axs.flat:
    ax.xaxis.offsetText.set_fontsize(FS)
    ax.yaxis.offsetText.set_fontsize(FS)
    ax.tick_params(axis='both',labelsize=FS)
    ax.grid()

plt.tight_layout()
plt.savefig(myExBprobe.directory+'E-,B-fields vs z at different (x,y).png',dpi=DPI,bbox_inches='tight',facecolor=fig.get_facecolor(),edgecolor=fig.get_edgecolor())
plt.show()
