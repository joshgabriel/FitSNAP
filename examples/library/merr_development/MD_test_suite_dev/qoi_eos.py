#!/home/jgabrie/SoftwareLibs/anaconda3/bin/python3
from mpi4py import MPI

import sys, string, os
import numpy as np
#from scipy.optimize import curve_fit
from scipy import optimize
#import matplotlib.pyplot as plt
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                               AutoMinorLocator, ScalarFormatter)
#from texcontext import latexContext
from lammps import lammps


# get current working dir
my_dir = os.path.join(os.getcwd())

# check input
argv = sys.argv
#if len(argv) != 3:
#  print("Syntax: ./fit_eos.py POSCAR_input ntypes")
#  sys.exit()
#dft_dat = sys.argv[1]
posfile = sys.argv[1]
ntypes = int(sys.argv[2])

#------------------------
# compute LAMMPS EOS
#------------------------

def lammps_eos(comp):
  #lmp = lammps()
  lmp = lammps(cmdargs=["-screen","none"])
  #lmp = lammps(name='mpi', cmdargs=["-screen","none"])
  # lmp = lammps(name='mpi')
  #block = """
  #        clear
  #        region  box block 0 2 0 2 0 2
  #        create_box 1 box
  #        create_atoms 1 single 1.0 1.0 ${zpos}
  #        """
  #lmp.commands_string(block)
  lmp.command("units        metal")
  lmp.command("atom_style   atomic")
  lmp.command("dimension    3")
  lmp.command("boundary     p p p")
  lmp.command("atom_modify  map array")

  lmp.command("region       box block 0.0 %g 0.0 %g 0.0 %g"
          % (L1[0],L2[1],L3[2]))
  lmp.command("create_box   3 box")


  lmp.command("mass         1 183.84")
  lmp.command("mass         2 91.224")
  lmp.command("mass         3 12.0107")

  lmp.command("include      in.snapWZrC")

  # rescale box
  lmp.command("change_box all x scale %g remap" % comp)
  lmp.command("change_box all y scale %g remap" % comp)
  lmp.command("change_box all z scale %g remap" % comp)

  #lmp.command("dump  1 all custom 1 dump_%s type x y z fx fy fz" % str(comp))

  lmp.command("run          0")

  lmp.command("variable     bxx equal lx")
  lmp.command("variable     byy equal ly")
  lmp.command("variable     vc equal vol")
  lmp.command("variable     na equal atoms")
  lmp.command("variable     etot equal etotal")
  lmp.command("variable     ptot equal press")

  # extract quantities

  N = lmp.get_natoms()
  press = lmp.extract_variable("ptot","all",0)
  eng = lmp.extract_variable("etot","all",0)
  vol = lmp.extract_variable("vc","all",0)
  nat = lmp.extract_variable("na","all",0)
  bxx = lmp.extract_variable("bxx","all",0)
  byy = lmp.extract_variable("byy","all",0)

  return eng, press, vol, nat

npoints = 10
v_lmp = np.empty(npoints)
p_lmp = np.empty(npoints)
e_lmp = np.empty(npoints)

# apply +/- 3% compression
clo = 0.97
chi = 1.03
# let's actually do 6%
clo = 0.94
chi = 1.06
aZrC = 4.72

for i in range(0, npoints):
  ci = clo + i*(chi-clo)/(npoints-1)
  e_lmp[i], p_lmp[i], v_lmp[i], nat = lammps_eos(ci)

  me = MPI.COMM_WORLD.Get_rank()
  nprocs = MPI.COMM_WORLD.Get_size()
  #print("Proc %d out of %d procs has" % (me,nprocs))
  #MPI.Finalize()
  e_lmp[i] /= nat
  v_lmp[i] /= nat
MPI.Finalize()
#----------------------------------
# compute bulk modulus calculations
#----------------------------------

# Birch-Murnaghan function and residuals
def bm_function(p,v):
  # p[0] => E0
  # p[1] => V0
  # p[2] => B0
  # p[3] => B0p
  Ev = (6.0-4.0*(p[1]/v)**(2.0/3.0))
  Ev *= ((p[1]/v)**(2.0/3.0)-1.0)**2.0
  Ev += p[3]*((p[1]/v)**(2.0/3.0)-1.0)**3.0
  Ev *= (9.0*p[1]*p[2]/16.0)
  Ev += p[0]
  return Ev

def residuals(p, v, e):
    return bm_function(p,v) - e

# perform BM fit on DFT data

#Ei = min(e_dft)
#Vi = min(ovol)
#p0 = [Ei,Vi,1.0,1.0]
#popt1, pcov1 = optimize.leastsq(residuals, p0, args=(ovol,e_dft))
#emin_dft = popt1[0]
#vmin_dft = popt1[1]
#B0_dft = popt1[2]*evA_to_gpa
# amin_dft = (vmin_dft)**(1.0/3.0)

# perform BM fit on SNAP data

Ei = min(e_lmp)
#Vi = min(v_lmp)
Vi = v_lmp[np.argmin(e_lmp)]
p0 = [Ei,Vi,1.0,1.0]
popt2, pcov2 = optimize.leastsq(residuals, p0, args=(v_lmp,e_lmp))
emin_sna = popt2[0]
vmin_sna = popt2[1]
B0_sna = popt2[2]*evA_to_gpa
# amin_sna = (vmin_sna)**(1.0/3.0)

B0_dft= 301.379
DE_B0=B0_dft-B0_sna

# get min energy, and corresponding (a,c)
# eopt = min(emin)
# index_opt = np.where(emin == eopt)
# index_opt = index_opt[0][0]
# aopt = amin[index_opt]
# vopt = vmin[index_opt]
# bopt = B0[index_opt]
print("   DFT   |  SNAP   |   DB")
print(" %g | %g | %g" % (B0_dft,B0_sna,DE_B0))
#print("### Optimal values: DFT | SNAP ###")
# print("### a = %g | %g" % (amin_dft,amin_sna))
#print("volume / atom = %g | %g" % (vmin_dft,vmin_sna))
#print("Bulk Modulus = %g | %g" % (B0_dft,B0_sna))
#print("Energy = %g | %g" % (emin_dft,emin_sna))
#print("Bulk Modulus difference")
#print(abs(B0_sna))

# display BM fits
