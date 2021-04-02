import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.patches as mpatches
font = { 'size'   : 15}
plt.rc('font', **font)

filename = 'plot_' + sys.argv[1].replace('.','_')
markersize = 10
colors = ['b','g','r','c','m','y']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

f = open(sys.argv[1], "r")

for line in f:
  if 'memroofs' in line:
    linesp = line.split()
    linesp = linesp[1:]
    smemroofs = [float(a) for a in linesp]
  if 'mem_roof_names' in line:
    linesp = line.strip().split("\'")
    linesp = filter(lambda a: (a != ' ') and (a != ''), linesp)
    smem_roof_name  = linesp[1:]
  if 'comproofs' in line:
    linesp = line.split()
    linesp = linesp[1:]
    scomproofs  = [float(a) for a in linesp]
  if 'comp_roof_names' in line:
    linesp = line.strip().split("\'")
    linesp = filter(lambda a: (a != ' ') and (a != ''), linesp)
    scomp_roof_name  = linesp[1:]
  if 'AI_HBM' in line:
    linesp = line.split()
    linesp = linesp[1:]
    AI_HBM = [float(a) for a in linesp]
  if 'AI_L2' in line:
    linesp = line.split()
    linesp = linesp[1:]
    AI_L2 = [float(a) for a in linesp]
  if 'AI_L1' in line:
    linesp = line.split()
    linesp = linesp[1:]
    AI_L1 = [float(a) for a in linesp]
  if 'FLOPS' in line:
    linesp = line.split()
    linesp = linesp[1:]
    FLOPS = [float(a) for a in linesp]
  if 'labels' in line:
    linesp=line.strip().split("\'")
    linesp = filter(lambda a: (a != ' ') and (a != ''), linesp)
    labels = linesp[1:]


fig = plt.figure(1,figsize=(10.67,6.6))
plt.clf()
ax = fig.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
ax.set_ylabel('Performance [GFLOP/sec]')
ax.set_title('Roofline Graph Kingspeak P100 GPU - <Sample Application>') #Change <Sample Application> to the name of your application 

nx = 10000
xmin = -0.65
xmax = 2
ymin = 500.0 
ymax = 13000

ax.set_xlim(10**xmin, 10**xmax)
ax.set_ylim(ymin, ymax)

ixx = int(nx*0.02)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

scomp_x_elbow = [] 
scomp_ix_elbow = [] 
smem_x_elbow = [] 
smem_ix_elbow = [] 

x = np.logspace(xmin,xmax,nx)
for roof in scomproofs:
    for ix in range(1,nx):
        if smemroofs[0] * x[ix] >= roof and smemroofs[0] * x[ix-1] < roof:
            scomp_x_elbow.append(x[ix-1])
            scomp_ix_elbow.append(ix-1)
            break


for roof in smemroofs:
    for ix in range(1,nx):
        if (scomproofs[0] <= roof * x[ix] and scomproofs[0] > roof * x[ix-1]):
            smem_x_elbow.append(x[ix-1])
            smem_ix_elbow.append(ix-1)
            break        

for i in range(0,len(scomproofs)):
    roof = scomproofs[i]
    y = np.ones(len(x)) * roof
    ax.plot(x[scomp_ix_elbow[i]:],y[scomp_ix_elbow[i]:],c='k',ls='-',lw='2')

for i in range(0,len(smemroofs)):
    roof = smemroofs[i]
    y = x * roof
    ax.plot(x[:smem_ix_elbow[i]+1],y[:smem_ix_elbow[i]+1],c='k',ls='-',lw='2')


marker_handles = list()

for i in range(0,len(AI_L1)):
  ax.plot(float(AI_L1[i]),float(FLOPS[i]),c=colors[0],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
for i in range(0,len(AI_L2)):
  ax.plot(float(AI_L2[i]),float(FLOPS[i]),c=colors[1],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
for i in range(0,len(AI_HBM)):
  ax.plot(float(AI_HBM[i]),float(FLOPS[i]),c=colors[i],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
  ax.legend(numpoints=1,loc='lower right',prop={'size':10})
  #marker_handles.append(ax.plot([],[],c='gray',marker=styles[i],linestyle='None',ms=markersize,label=labels[i])[0]) 



for roof in scomproofs:
    ax.text(x[-ixx],roof,
            scomp_roof_name[scomproofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GFLOP/s',
            horizontalalignment='right',
            verticalalignment='bottom')

for roof in smemroofs:
    ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0]) 
                                 * fig.get_size_inches()[1]/fig.get_size_inches()[0] )
    if x[ixx]*roof >ymin:
        ax.text(x[ixx],x[ixx]*roof*(1+0.25*np.sin(ang)**2),
            smem_roof_name[smemroofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=180/np.pi*ang)
    else:
        ymin_ix_elbow=list()
        ymin_x_elbow=list()
        for ix in range(1,nx):
            if (ymin <= roof * x[ix] and ymin > roof * x[ix-1]):
                ymin_x_elbow.append(x[ix-1])
                ymin_ix_elbow.append(ix-1)
                break
        ax.text(x[ixx+ymin_ix_elbow[0]],x[ixx+ymin_ix_elbow[0]]*roof*(1+0.25*np.sin(ang)**2),
            smem_roof_name[smemroofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=180/np.pi*ang)


patch_handles = list()
for i in range(0,len(smem_roof_name)):
    patch_handles.append(mpatches.Patch(color=colors[i],label = smem_roof_name[i]))

plt.grid(b=True, which='both', axis='both', linestyle='-', color='0.6')
plt.savefig(filename+'.png')
