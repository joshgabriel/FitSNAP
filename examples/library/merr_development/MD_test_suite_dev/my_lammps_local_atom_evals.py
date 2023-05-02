"""# Demonstrate how to evaluate a series of snapshots with different snap coefficients to get uncertainty estimates
# plot out local energy uncertainty and force magnitude uncertainty for each atom
# plot statistics of distance matrix of local atomic environment for highest and lowest uncertainty atoms
"""
#!/home/jgabrie/SoftwareLibs/anaconda3/bin/python3
from mpi4py import MPI
import matplotlib.pyplot as plt
import os, sys, linecache
import numpy as np
from glob import glob
from ase.io import read
from ase.io.lammpsrun import read_lammps_dump
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
import time 
import json
import lammps

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

def post_process_dumps(f):
        t1 = time.time()
        data = read(f)
        data_dict = data.todict()
        # extract local atom energy, force magnitude
        emags, fmags= \
            list(data_dict['c_eatom']), [np.linalg.norm([data_dict[i][num] for i in ['fx','fy','fz']]) \
            for num in range(len(data_dict['c_eatom']))]
        struct = AseAtomsAdaptor.get_structure(data)
        #site_dists=struct.get_all_neighbors_py(r=5.0)
        #print (len(site_dists))
        t2 = time.time()
        #print ("Processed in {}...".format(t2-t1))
        return emags, fmags, struct

def process_local_dist_stats(structure, index):
        sites = struct.get_neighbors(index)
        # get boxed carved structure
        struct = None 
        # isolate environments, compute their distance matrices

        try:
           avg_dist = np.mean(np.ravel(struct.distance_matrix))
        except:
           avg_dist = 10.0
        return avg_dist

before_loading =\
"""
units		metal
boundary	p p p
read_data       lmp.data

mass    1 180.94

include Ta_pot.mod
"""

after_loading =\
"""
variable latpar2 equal lx/6
compute eatom all pe/atom
compute Sumenergy all reduce sum c_eatom

velocity all create 10 34583
timestep 5E-4
fix 2 all npt temp 10 293 0.05 iso 0.0 0.0 0.5

thermo_style custom step dt time pe ke etotal c_Sumenergy press vol v_latpar2
thermo 1
dump snap all cfg 1 dump.single.*.cfg mass type xs ys zs c_eatom fx fy fz

neigh_modify once no every 1 delay 0 check yes

run 0
"""

if __name__ == "__main__":
   coeff_store = "/home/jgabrie/Runs/Ta_pub_examination/MERR_all_uncorr/PCE_Explorations/"
   coeff_list = [f.split('.')[-1] for f in glob("/home/jgabrie/Runs/Ta_pub_examination/MERR_all_uncorr/PCE_Explorations/*.snapcoeff.*")][0:100] 
   snapshot_results = {}

   for sn in [2+ 1000*n for n in list(range(60))]+[60000]:
       t1 = time.time()
       coeff_results_list = {c:None for c in coeff_list}
       f = "Dumps/dump.config.{}.cfg".format(sn)
       ase_data = read(f)
       struct = AseAtomsAdaptor.get_structure(ase_data)
       Ta_struct = Structure(struct.lattice, ['Ta' for s in struct.species],\
               struct.cart_coords,coords_are_cartesian=True)
       LammpsData.from_structure(Ta_struct,atom_style='atomic').write_file("lmp.data")
       
       for coeff in coeff_list:

           print (sn, coeff)
           os.system("cp {0} Ta_pot.snapcoeff".format(coeff_store+os.sep+"Ta_pot.snapcoeff."+coeff))

           try:
               lmp = lammps.lammps(cmdargs=["-screen","none"]) #cmdargs=['-echo','both'])

               lmp.commands_string(before_loading)

               # run the simulation
               lmp.commands_string(after_loading)
               lx = lmp.extract_variable("lx","all",0)
               #MPI.Finalize()
               Emags_coeff, Fmags_coeff, struct_coeff = post_process_dumps("dump.single.0.cfg")
               coeff_results_list[coeff] = {"Coeff":coeff,"Emags":Emags_coeff,"Fmags":Fmags_coeff,\
                   "Structure":struct_coeff.as_dict()}
           except:
               print (print_exception())#"Exception")
               coeff_results_list[coeff] = None
       snapshot_results[f] = coeff_results_list
       t2 = time.time()
       print ("Done one snapshot in ",t2-t1)
   MPI.Finalize()
   with open("Results.json","w") as wf:
       json.dump(snapshot_results,wf)
   wf.close()

   # postprocessing towards plots

   #d['Dumps/dump.config.2.cfg']['0005']['Emags'][0
   max_uq_energies = []
   max_uq_energies_loc = []
   max_uq_forces = []
   max_uq_forces_loc = []

   min_uq_energies = []
   min_uq_energies_loc = []
   min_uq_forces = []
   min_uq_forces_loc = []

   for num in range(432):
       energy_uq = \
        [np.std([snapshot_results[snap][k]['Emags'][num] for k in coeff_list]) \
        for snap in snapshot_results]
       max_uq_loc_energy = np.argmax(energy_uq)
       max_uq_energy = max(energy_uq)
       min_uq_loc_energy = np.argmin(energy_uq)
       min_uq_energy = min(energy_uq)

       force_uq = \
        [np.std([snapshot_results[snap][k]['Fmags'][num] for k in coeff_list]) \
        for snap in snapshot_results]

       max_uq_loc_force = np.argmax(force_uq)
       max_uq_force = max(force_uq)
       min_uq_loc_force = np.argmin(force_uq)
       min_uq_force = min(force_uq)

       max_uq_energies.append(max_uq_energy)
       min_uq_energies.append(min_uq_energy)
       max_uq_energies_loc.append(max_uq_loc_energy)
       min_uq_energies_loc.append(min_uq_loc_energy)


       max_uq_forces.append(max_uq_force)
       min_uq_forces.append(min_uq_force)
       max_uq_forces_loc.append(max_uq_loc_force)
       min_uq_forces_loc.append(min_uq_loc_force)

   # collated results and plots
   CollResults= {"UQ.LocalEnergy":{"Max":\
                      {"Value":max(max_uq_energies),
                      "AtomID":float(np.argmax(max_uq_energies)),
                      "SnapShot":float(max_uq_energies_loc[np.argmax(max_uq_energies)])
                      },
                      "Min":\
                      {"Value":min(min_uq_energies),
                      "AtomID":float(np.argmin(min_uq_energies)),
                      "SnapShot":float(min_uq_energies_loc[np.argmin(min_uq_energies)])
                      }
                    },
    "UQ.LocalForce":{"Max":\
                      {"Value":max(max_uq_forces),
                      "AtomID":float(np.argmax(max_uq_forces)),
                      "SnapShot":float(max_uq_forces_loc[np.argmax(max_uq_forces)])
                      },
                      "Min":\
                      {"Value":min(min_uq_forces),
                      "AtomID":float(np.argmin(min_uq_forces)),
                      "SnapShot":float(min_uq_forces_loc[np.argmin(min_uq_forces)])
                      }
                    }
   }

  
   with open("CollResults.json","w") as wf:
       json.dump(CollResults,wf)
   wf.close()

   atom_indices = [int(CollResults["UQ.LocalEnergy"]["Min"]["AtomID"]), int(CollResults["UQ.LocalEnergy"]["Max"]["AtomID"])]
   labs = ["Min Energy UQ @ atom {0}".format(atom_indices[0]),\
           "Max Energy UQ @ atom {0}".format(atom_indices[1])]

   for num,at in enumerate(atom_indices):
       energy_uq = \
        [np.std([snapshot_results[snap][k]['Emags'][at] for k in coeff_list]) \
        for snap in snapshot_results]
       plt.plot(energy_uq,label=labs[num])

   plt.legend()
   plt.xlabel("Snapshot")
   plt.ylabel("Energy (eV)")
   plt.savefig("Energy_UQ.png")
   plt.close()

   atom_indices = [int(CollResults["UQ.LocalForce"]["Min"]["AtomID"]), int(CollResults["UQ.LocalForce"]["Max"]["AtomID"])]
   labs = ["Min Force UQ @ atom {0}".format(atom_indices[0]),\
           "Max Force UQ @ atom {0}".format(atom_indices[1])]


   for num,at in enumerate(atom_indices):
       force_uq = \
        [np.std([snapshot_results[snap][k]['Fmags'][at] for k in coeff_list]) \
        for snap in snapshot_results]
       plt.plot(force_uq,label=labs[num])

   plt.legend()
   plt.xlabel("Snapshot")
   plt.ylabel("Force (eV/A)")
   plt.savefig("Force_UQ.png")

