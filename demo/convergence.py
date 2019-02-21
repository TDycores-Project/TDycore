"""blah
"""
import subprocess
import pylab as plt
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', dest="dim", metavar='dim', type=int,default=2)
parser.add_argument('-p', dest="problem", metavar='problem', type=int,default=3)
parser.add_argument('-m', dest="method", metavar='method', type=str,default="wy")
args = parser.parse_args()

solver = "-pc_type lu -pc_factor_mat_solver_type umfpack"
method = "-tdy_method %s" % args.method
problem = "-problem %d" % args.problem

def _buildTriangle(rate,offset,h,E):
    dh = np.log10(h[0])-np.log10(h[-1])
    x  =    [10**(np.log10(h[-1])+ offset      *dh)]
    x.append(10**(np.log10(h[-1])+(offset+0.15)*dh))
    x.append(x[-1])
    x.append(x[ 0])
    y = [E[-1],E[-1]]
    y.append( 10**(np.log10(E[-1]) + rate*(np.log10(x[1])-np.log10(x[0]))))
    y.append(E[-1])
    return x,y

d = args.dim
N = np.asarray([8,16,32,64,128]) #,256,512])
if d == 3: N = np.asarray([4,8,16,32])

h = 1./N
E = np.zeros((N.size,3))
for i,n in enumerate(N):
    ref = "" if i == 0 else " -dm_refine %d" % i
    process = subprocess.Popen("./steady -dim %d -N %d %s %s %s %s" % (d,N[0],ref,solver,method,problem),
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    output,errors = process.communicate()
    E[i] = [float(v) for v in output.split()]
    print("%3d  %.6e %.6e %.6e" % (n,E[i,0],E[i,1],E[i,2]))
s = 2
print("rate = %.2f  %.2f  %.2f" % (np.polyfit(np.log10(h[s:]),np.log10(E[s:,0]),1)[0],
                                   np.polyfit(np.log10(h[s:]),np.log10(E[s:,1]),1)[0],
                                   np.polyfit(np.log10(h[s:]),np.log10(E[s:,2]),1)[0]))

pad = 0.05
lbl = 0.19
dh  = np.log10(h[0])-np.log10(h[-1])
dh *= (1. + 2*pad + lbl)
hL  = 10**(np.log10(h[-1]) -  pad*dh)
hR  = 10**(np.log10(h[ 0]) + (pad+lbl)*dh)
h0  = 10**(np.log10(h[ 0]) + 0.02*dh)

fig,ax = plt.subplots(tight_layout=True)
ax.loglog(h,E[:,0],'-o',ms=4)
ax.loglog(h,E[:,1],'-^',ms=4)
ax.loglog(h,E[:,2],'-s',ms=4)
tx,ty = _buildTriangle(2.0,0.05,h,E[:,0])
ax.loglog(tx,ty,'-k')
ax.text(tx[1],ty[1],"2 ",ha="right",va="bottom")
ax.text(h0,E[0,0],r"$|||p-p_h|||$")
ax.text(h0,E[0,1],r"$|||\mathbf{u}-\mathbf{u}_h|||$")
ax.text(h0,E[0,2],r"$|||\nabla\cdot\left(\mathbf{u}-\mathbf{u}_h\right)|||$")
ax.set_xlim(hL,hR)
ax.set_xlabel("Mesh size, $h$")
ax.set_ylabel("Norm of Error")
fig.savefig("convergence%dD.png" % d)
