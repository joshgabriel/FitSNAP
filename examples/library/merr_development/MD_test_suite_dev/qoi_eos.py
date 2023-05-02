#!/home/jgabrie/SoftwareLibs/anaconda3/bin/python3
from mpi4py import MPI
import numpy as np
from scipy import optimize
from lammps import lammps
import matplotlib.pyplot as plt
import os, sys, linecache
from glob import glob
from ase.io import read
from ase.io.lammpsrun import read_lammps_dump
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
import time 
import json



before_loading =\
"""
units   metal
boundary  p p p
read_data       lmp.data

mass    1 180.94

include pot.mod
"""

after_loading =\
"""
variable latpar2 equal lx/6
compute eatom all pe/atom
compute Sumenergy all reduce sum c_eatom

velocity all create 10 34583
timestep 5E-4
fix 1 all nvt temp 10 293 0.05 iso 0.0 0.0 0.5

thermo_style custom step temp time pe ke etotal c_Sumenergy c_eatom press vol v_latpar2
thermo 1

neigh_modify once no every 1 delay 0 check yes

run 60000
"""


def print_exception():
    """
    Error exception catching function for debugging
    can be a very useful tool for a developer
    move to utils and activate when debug mode is on
    """
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    #logging.warning('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
    #                                                   line.strip(), exc_obj))
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
                                                       line.strip(), exc_obj))



def lammps_npt_finite_T_vol(struct,scale):
    lmp = lammps(cmdargs=["-screen","none"])

    # write out data file for the simulation
    struct_out = Structure(scale*struct.lattice, struct.species,\
                 struct.frac_coords)
    LammpsData.from_structure(struct_out,atom_style='atomic').write_file("lmp.data")
    lmp.commands_string(before_loading)
    # run the simulation
    lmp.commands_string(after_loading)

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



if __name__ == "__main__":
    # get current working dir
    my_dir = os.path.join(os.getcwd())

    # check input
    argv = sys.argv
 
    posfile = sys.argv[1]
    ntypes = int(sys.argv[2])

    struct = Poscar.from_file(posfile).structure
    struct.make_supercell([3,3,3])

    #------------------------
    # compute LAMMPS EOS
    #------------------------


    npoints = 10
    v_lmp = np.empty(npoints)
    p_lmp = np.empty(npoints)
    e_lmp = np.empty(npoints)

    # let's actually do 6%
    clo = 0.94
    chi = 1.06
  
    for i in range(0, npoints):
      ci = clo + i*(chi-clo)/(npoints-1)
      e_lmp[i], p_lmp[i], v_lmp[i], nat = lammps_npt_finite_T_vol(struct,ci)

      me = MPI.COMM_WORLD.Get_rank()
      nprocs = MPI.COMM_WORLD.Get_size()
      e_lmp[i] /= nat
      v_lmp[i] /= nat
    MPI.Finalize()
    #----------------------------------
    # compute bulk modulus calculations
    #----------------------------------

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


    with open("EOS_Results.txt","w") as wf:
      wf.write("   DFT   |  SNAP   |   DB")
      wf.write(" %g | %g | %g" % (B0_dft,B0_sna,DE_B0))
    wf.close()
