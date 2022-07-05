from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


class LammpsSnap(LammpsBase):

    def __init__(self, name):
        super().__init__(name)
        self._data = {}
        self._i = 0
        self._lmp = None
        pt.check_lammps()

    def get_width(self):
        num_types = config.sections["BISPECTRUM"].numtypes
        a_width = config.sections["BISPECTRUM"].ncoeff * num_types
        if not config.sections["BISPECTRUM"].bzeroflag:
            a_width += num_types
        return a_width

    def _prepare_lammps(self):
        self._set_structure()
        # this is super clean when there is only one value per key, needs reworking
        #        self._set_variables(**_lammps_variables(config.sections["BISPECTRUM"].__dict__))

        # Needs reworking when lammps will accept variable 2J
        self._lmp.command(f"variable twojmax equal {max(config.sections['BISPECTRUM'].twojmax)}")
        self._lmp.command(f"variable rcutfac equal {config.sections['BISPECTRUM'].rcutfac}")
        self._lmp.command(f"variable rfac0 equal {config.sections['BISPECTRUM'].rfac0}")
        #        self._lmp.command(f"variable rmin0 equal {config.sections['BISPECTRUM'].rmin0}")

        for i, j in enumerate(config.sections["BISPECTRUM"].wj):
            self._lmp.command(f"variable wj{i + 1} equal {j}")

        for i, j in enumerate(config.sections["BISPECTRUM"].radelem):
            self._lmp.command(f"variable radelem{i + 1} equal {j}")

        for line in config.sections["REFERENCE"].lmp_pairdecl:
            self._lmp.command(line.lower())

        self._set_computes()
        self._set_neighbor_list()

    def _set_box(self):
        self._set_box_helper(numtypes=config.sections['BISPECTRUM'].numtypes)

    def _create_atoms(self):
        self._create_atoms_helper(type_mapping=config.sections["BISPECTRUM"].type_mapping)

    def _set_computes(self):
        numtypes = config.sections['BISPECTRUM'].numtypes
        radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        kw_options = {
            k: config.sections["BISPECTRUM"].__dict__[v]
            for k, v in
            {
                "rmin0": "rmin0",
                "bzeroflag": "bzeroflag",
                "quadraticflag": "quadraticflag",
                "switchflag": "switchflag",
                "chem": "chemflag",
                "bnormflag": "bnormflag",
                "wselfallflag": "wselfallflag",
                "bikflag": "bikflag",
#                "switchinnerflag": "switchinnerflag",
#                "sinner": "sinner",
#                "dinner": "dinner",
            }.items()
            if v in config.sections["BISPECTRUM"].__dict__
        }
        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        if kw_options["bikflag"] == 0:
            kw_options.pop("bikflag")
        kw_options["rmin0"] = config.sections["BISPECTRUM"].rmin0
        kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)

    def _collect_lammps(self):

        num_atoms = self._data["NumAtoms"]
        num_types = config.sections['BISPECTRUM'].numtypes
        n_coeff = config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        bik_rows = 1
        if config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_snap = nrows_energy + nrows_force + nrows_virial
        ncols_bispectrum = n_coeff * num_types
        ncols_reference = 1
        ncols_snap = ncols_bispectrum + ncols_reference
        # index = pt.fitsnap_dict['a_indices'][self._i]
        index = self.shared_index
        dindex = self.distributed_index

        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))

        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))
        irow = 0
        bik_rows = 1
        if config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow:irow+bik_rows, :ncols_bispectrum] / num_atoms

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            EPS = 1.0e-10
            b000sum0 = 0.0
            nstride = n_coeff
            if not config.sections['BISPECTRUM'].bikflag:
                if not config.sections["BISPECTRUM"].bzeroflag:
                    b000sum0 = 1.0
                if config.sections["BISPECTRUM"].chemflag:
                    nstride //= num_types*num_types*num_types
                    if config.sections["BISPECTRUM"].wselfallflag:
                        b000sum0 *= num_types*num_types*num_types
                b000sum = sum(b_sum_temp[0, ::nstride])
                if abs(b000sum - b000sum0) < EPS:
                    print("WARNING: Configuration has no SNAP neighbors")

            if not config.sections["BISPECTRUM"].bzeroflag:
                if config.sections['BISPECTRUM'].bikflag:
                    raise NotImplementedError("per atom energy is not implemented without bzeroflag")
                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.zeros((num_types, 1))
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[config.sections["BISPECTRUM"].type_mapping[atom]-1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                b_sum_temp.shape = (num_types * n_coeff + num_types)

            pt.shared_arrays['a'].array[index:index+bik_rows] = \
                b_sum_temp * config.sections["BISPECTRUM"].blank2J[np.newaxis, :]
            ref_energy = lmp_snap[irow, icolref]
            pt.shared_arrays['b'].array[index:index+bik_rows] = 0.0
            pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            pt.shared_arrays['w'].array[index] = self._data["eweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not config.sections["BISPECTRUM"].bzeroflag:
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index:index+nrows_force] = \
                np.matmul(db_atom_temp, np.diag(config.sections["BISPECTRUM"].blank2J))
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            pt.shared_arrays['b'].array[index:index+nrows_force] = \
                self._data["Forces"].ravel() - ref_forces
            pt.shared_arrays['w'].array[index:index+nrows_force] = \
                self._data["fweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not config.sections["BISPECTRUM"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index:index+ndim_virial] = \
                np.matmul(vb_sum_temp, np.diag(config.sections["BISPECTRUM"].blank2J))
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            pt.shared_arrays['w'].array[index:index+ndim_virial] = \
                self._data["vweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
            pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            index += ndim_virial
            dindex += ndim_virial

        length = dindex - self.distributed_index
        pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
        pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
        pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
        self.shared_index = index
        self.distributed_index = dindex
