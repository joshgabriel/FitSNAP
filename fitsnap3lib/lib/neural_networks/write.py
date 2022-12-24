import numpy as np
import torch
import math

def calc_n_params(model):
    """
    Returns the sum of two decimal numbers in binary digits.

        Parameters:
                model (torch.nn.Module): Network model that maps descriptors to a per atom attribute

        Returns:
                n_params (int): Number of NN model parameters
    """
    return sum(p.nelement() for p in model.parameters())

class TorchWrapper(torch.nn.Module):
    """
    A class to wrap Modules to ensure lammps mliap compatability.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        Network model that maps descriptors to a per atom attribute

    device : torch.nn.Module (None)
        Accelerator device

    dtype : torch.dtype (torch.float64)
        Dtype to use on device

    n_params : torch.nn.Module (None)
        Number of NN model parameters

    n_descriptors : int
        Max number of per atom descriptors

    n_elements : int
        Max number of elements


    Methods
    -------
    forward(descriptors, elems):
        Feeds descriptors to network model to produce per atom energies and forces.
    """

    def __init__(self, model, n_descriptors, n_elements, n_params=None, device=None, dtype=torch.float64):
        """
        Constructs all the necessary attributes for the network module.

        Parameters
        ----------
            model : torch.nn.Module
                Network model that maps descriptors to a per atom attribute

            n_descriptors : int
                Max number of per atom descriptors

            n_elements : int
                Max number of elements

            n_params : torch.nn.Module (None)
                Number of NN model parameters
            
            device : torch.nn.Module (None)
                Accelerator device
            
            dtype : torch.dtype (torch.float64)
                Dtype to use on device
        """

        super().__init__()

        self.model = model
        self.device = device
        self.dtype = dtype

        # Put model on device and convert to dtype
        self.to(self.dtype)
        self.to(self.device)

        if n_params is None:
            n_params = calc_n_params(model)

        self.n_params = n_params
        self.n_descriptors = n_descriptors
        self.n_elements = n_elements

    def forward(self, elems, descriptors, beta, energy):
        """
        Takes element types and descriptors calculated via lammps and
        calculates the per atom energies and forces.

        Parameters
        ----------
        elems : numpy.array
            Per atom element types

        descriptors : numpy.array
            Per atom descriptors

        beta : numpy.array
            Expired beta array to be filled with new betas

        energy : numpy.array
            Expired per atom energy array to be filled with new per atom energy
            (Note: This is a pointer to the lammps per atom energies)


        Returns
        -------
        None
        """

        descriptors = torch.from_numpy(descriptors).to(dtype=self.dtype, device=self.device).requires_grad_(True)
        elems = torch.from_numpy(elems).to(dtype=torch.long, device=self.device) - 1

        with torch.autograd.enable_grad():

            energy_nn = self.model(descriptors, elems)
            if energy_nn.ndim > 1:
                energy_nn = energy_nn.flatten()

            beta_nn = torch.autograd.grad(energy_nn.sum(), descriptors)[0]

        beta[:] = beta_nn.detach().cpu().numpy().astype(np.float64)
        energy[:] = energy_nn.detach().cpu().numpy().astype(np.float64)


class IgnoreElems(torch.nn.Module):
    """
    A class to represent a NN model agnostic of element typing.

    ...

    Attributes
    ----------
    subnet : torch.nn.Module
        Network model that maps descriptors to a per atom attribute

    Methods
    -------
    forward(descriptors, elems):
        Feeds descriptors to network model
    """

    def __init__(self, subnet):
        """
        Constructs all the necessary attributes for the network module.

        Parameters
        ----------
            subnet : torch.nn.Module
                Network model that maps descriptors to a per atom attribute
        """

        super().__init__()
        self.subnet = subnet

    def forward(self, descriptors, elems=None):
        """
        Feeds descriptors to network model

        Parameters
        ----------
        descriptors : torch.tensor
            Per atom descriptors

        elems : torch.tensor
            Per atom element types

        Returns
        -------
        self.subnet(descriptors) : torch.tensor
            Per atom attribute computed by the network model
        """

        return self.subnet(descriptors)


class UnpackElems(torch.nn.Module):
    """
    A class to represent a NN model pseudo-agnostic of element typing for
    systems with multiple element typings.

    ...

    Attributes
    ----------
    subnet : torch.nn.Module
        Network model that maps descriptors to a per atom attribute

    n_types : int
        Number of atom types used in training the NN model.

    Methods
    -------
    forward(descriptors, elems):
        Feeds descriptors to network model after adding zeros into
        descriptor columns relating to different atom types
    """

    def __init__(self, subnet, n_types):
        """
        Constructs all the necessary attributes for the network module.

        Parameters
        ----------
            subnet : torch.nn.Module
                Network model that maps descriptors to a per atom attribute.

            n_types : int
                Number of atom types used in training the NN model.
        """
        super().__init__()
        self.subnet = subnet
        self.n_types = n_types

    def forward(self, descriptors, elems):
        """
        Feeds descriptors to network model after adding zeros into
        descriptor columns relating to different atom types

        Parameters
        ----------
        descriptors : torch.tensor
            Per atom descriptors

        elems : torch.tensor
            Per atom element types

        Returns
        -------
        self.subnet(descriptors) : torch.tensor
            Per atom attribute computed by the network model
        """

        unpacked_descriptors = torch.zeros(elems.shape[0], self.n_types, descriptors.shape[1], dtype=torch.float64)
        for i, ind in enumerate(elems):
            unpacked_descriptors[i, ind, :] = descriptors[i]
        return self.subnet(torch.reshape(unpacked_descriptors, (elems.shape[0], -1)), elems)


class ElemwiseModels(torch.nn.Module):
    """
    A class to represent a NN model dependent on element typing.

    ...

    Attributes
    ----------
    subnets : list of torch.nn.Modules
        Per element type network models that maps per element type
        descriptors to a per atom attribute.

    n_types : int
        Number of atom types used in training the NN model.

    Methods
    -------
    forward(descriptors, elems):
        Feeds descriptors to network model after adding zeros into
        descriptor columns relating to different atom types
    """

    def __init__(self, subnets, n_types):
        """
        Constructs all the necessary attributes for the network module.

        Parameters
        ----------
            subnets : list of torch.nn.Modules
                Per element type network models that maps per element
                type descriptors to a per atom attribute.

            n_types : int
                Number of atom types used in training the NN model.
        """

        super().__init__()
        self.subnets = subnets
        self.n_types = n_types

    def forward(self, descriptors, elems, dtype=torch.float64):
        """
        Feeds descriptors to network model after adding zeros into
        descriptor columns relating to different atom types

        Parameters
        ----------
        descriptors : torch.tensor
            Per atom descriptors

        elems : torch.tensor
            Per atom element types

        Returns
        -------
        self.subnets(descriptors) : torch.tensor
            Per atom attribute computed by the network model
        """

        self.dtype=dtype
        self.to(self.dtype)

        per_atom_attributes = torch.zeros(elems.size(dim=0), dtype=self.dtype)
        given_elems, elem_indices = torch.unique(elems, return_inverse=True)
        for i, elem in enumerate(given_elems):
            self.subnets[elem].to(self.dtype) 
            per_atom_attributes[elem_indices == i] = self.subnets[elem](descriptors[elem_indices == i]).flatten()
        return per_atom_attributes

class PairNN(torch.nn.Module):
    """
    A class to wrap Modules to ensure lammps mliap compatability.
    """

    def __init__(self, model, n_descriptors, n_elements, n_params=None, device=None, dtype=torch.float64):
        """
        Constructs all the necessary attributes for the network module.
        """

        super().__init__()

        print("^^^^^ write.py PairNN init")

        self.model = model
        self.device = device
        self.dtype = dtype

        # Put model on device and convert to dtype
        self.to(self.dtype)
        self.to(self.device)

        if n_params is None:
            n_params = calc_n_params(model)

        self.n_params = n_params
        self.n_descriptors = n_descriptors
        self.n_elements = n_elements

        # Parameters of model that will be used by LAMMPS:

        self.cutoff = 3.0
        self.num_radial_descriptors = 5
        self.num_3body_descriptors = 12

        # Parameters used by g3b:

        self.pi = torch.tensor(math.pi)
        self.eta = 4.
        self.mu = torch.linspace(-1,1,self.num_3body_descriptors)

        # trying to make a bessel object doesn't work:
        #from fitsnap3lib.lib.neural_networks.descriptors.bessel import Bessel
        #self.bessel = Bessel(self.num_radial_descriptors, self.cutoff) # Bessel object provides functions to calculate descriptors


    def cutoff_function(self, r, numpy_bool=False):
        """
        Calculate cutoff function for all rij.

        Args:
            r (torch.Tensor.float): Array of pairwise distances with size (npairs, 1)

        Returns:
            function (torch.Tensor.float): Cutoff function values for all rij. 
        """

        rmin = 3.5

        mask = r > rmin
        if (r.dtype==torch.float64):
            function = torch.empty(r.size()).double() # need double if doing FD test
        else:
            function = torch.empty(r.size())

        c = self.cutoff
        pi = torch.tensor(math.pi)

        function[mask] = 0.5 + 0.5*torch.cos(pi*(r[mask]-rmin)/(c-rmin))
        function[~mask] = 1.0

        return function

    def cutoff_function_g3b(self, rij):
        """
        Calculate cutoff function for all rij, used for g3b descriptors.

        Args:
            rij (:obj:`torch.Tensor`): Pairwise distances of all pairs in this batch, size 
                                        (num_neighs, 1) where `num_neighs` is number of neighbors 
                                        for all atoms in this batch.
        """

        c = self.cutoff
        function = 0.5 + 0.5*torch.cos(self.pi*(rij-0)/(c-0))

        return function[:,0]

    def calculate_bessel(self, rij, cutoff_functions, n):
        """
        Calculate a specific radial bessel function `n` for all pairs.

        Args:
            rij (torch.Tensor.float): Pairwise distance tensor with size (num_neigh, 1).
            n (torch.Tensor.float): Integer in float form representing Bessel radial parameter n.

        Returns:
            rbf (torch.Tensor.float): Radial Bessel function for base n with size (num_neigh, 1).
        """

        c = self.cutoff
        pi = torch.tensor(math.pi)
        two_over_c = torch.tensor(2./c)
        rbf = torch.div(torch.sqrt(two_over_c)*torch.sin(((n*pi)/c)*rij), rij)*cutoff_functions     

        return rbf

    def radial_bessel_basis(self, rij, cutoff_functions, numpy_bool = False):
        """
        Calculate radial Bessel basis functions.

        Args:
            x (torch.Tensor.float): Array of positions for this batch.
            neighlist (torch.Tensor.long): Sparse neighlist for this batch.
            unique_i (torch.Tensor.long): Atoms i for all atoms in this batch indexed starting 
                from 0 to (natoms_batch-1).
            xneigh (torch.Tensor.float): Positions of neighbors corresponding to indices j in 
                the neighbor list.

        Returns:
            basis (torch.Tensor.float): Concatenated tensor of Bessel functions for all pairs 
                with size (num_neigh, num_descriptors)
        """

        basis = torch.cat([self.calculate_bessel(rij, cutoff_functions, n) for n in range(1,self.num_radial_descriptors+1)], dim=1)

        return basis

    def calculate_g3b(self, rij, diff_norm, unique_i):
        """
        Calculate Gaussian 3body descriptors for all pairs. In the following discussion, :code:`num_neighs` 
        is the total number of neighbors in the entire batch, also equivalent to the total 
        number of pairs.

        Args:
            rij (:obj:`torch.tensor`): Pairwise distances of all pairs in this batch, size 
                                        (num_neighs, 1) where :code:`num_neighs` is number of neighbors 
                                        for all atoms in this batch
            diff_norm (:obj:`torch.tensor`): Pairwise normalized dispalcements between all pairs 
                                              in this batch, size (num_neighs, 3)
            unique_i (:obj:`torch.long`): Indices of atoms i in this batch, size (num_neighs)

        Returns:
            :obj:`torch.tensor`: Tensor of size (num_neighs, num_3body_descriptors)
        """

        # cutoff function for all pairs, size (num_neigh)

        fcrik = self.cutoff_function_g3b(rij) #.flatten()
        print(np.shape(rij))
        print(f"Max fcrik: {torch.max(fcrik)}")

        ui = unique_i.unique()

        descriptors_3body = torch.cat([torch.sum(
                                    torch.exp(-1.0*self.eta
                                        * (torch.mm(diff_norm[unique_i==i], 
                                            torch.transpose(diff_norm[unique_i==i],0,1)).fill_diagonal_(0)[:,:,None]
                                        -self.mu)**2) 
                                    * fcrik[unique_i==i][:,None], 
                                  dim=1)
                                for i in ui],
                                dim=0)

        return descriptors_3body

    def forward(self, elems, descriptors, beta, energy, rij, unique_i, unique_j,\
                tag_i, tag_j):
        """
        Takes element types and descriptors calculated via lammps and
        calculates the per atom energies and forces.

        Parameters
        ----------
        elems : numpy.array
            Per atom element types

        descriptors : numpy.array
            Per atom descriptors

        beta : numpy.array
            Expired beta array to be filled with new betas

        energy : numpy.array
            Expired per atom energy array to be filled with new per atom energy
            (Note: This is a pointer to the lammps per atom energies)

        rij : numpy.array
            Vector of pairwise displacements xj - xi

        unique_i (:obj:`torch.Tensor.long`): Atoms i for all pairs in this config.

        tag_i and tag_j : Tags (actually tag-1) for atoms in all pairs.

        Returns
        -------
        None
        """

        print("^^^^^ write.py pairnn forward")

        #print("^^^^^ write.py rij:")
        #print(rij)
        #print(self.cutoff)

        rij = torch.from_numpy(rij).to(dtype=self.dtype, device=self.device).requires_grad_(True)
        unique_i = torch.from_numpy(unique_i).to(dtype=torch.long, device=self.device) #.requires_grad_(True)
        unique_j = torch.from_numpy(unique_j).to(dtype=torch.long, device=self.device) #.requires_grad_(True)
        tag_i = torch.from_numpy(tag_i).to(dtype=torch.long, device=self.device) #.requires_grad_(True)
        tag_j = torch.from_numpy(tag_j).to(dtype=torch.long, device=self.device) #.requires_grad_(True)

        print(unique_i)
        print(tag_i)
        print(unique_j)
        print(tag_j)
        #assert(False)

        diff_norm = torch.nn.functional.normalize(rij, dim=1) # need for g3b
                                                              # size (npairs, 3)
        distance_ij = torch.linalg.norm(rij, dim=1).unsqueeze(1)  # need for cutoff and various other functions
                                                                  # size (npairs, 1)

        #print(distance_ij)

        #print(unique_i)
        #neighlist = torch.cat((unique_i, unique_j), dim=1)

        maxr = torch.max(distance_ij)
        print(f"Max pairwise distance: {maxr}")

        # Calculate cutoff functions once for pairwise terms here, because we use the same cutoff 
        # function for both radial basis and eij.
        # Values will be 1.0 if r < rmin

        cutoff_functions = self.cutoff_function(distance_ij) # size (npairs, 1)

        # calculate Bessel radial basis

        rbf = self.radial_bessel_basis(distance_ij, cutoff_functions)
        assert(rbf.size()[0] == rij.size()[0])

        print(f"Max RBF: {torch.max(rbf)}")

        # calculate 3 body descriptors 

        descriptors_3body = self.calculate_g3b(distance_ij, diff_norm, unique_i)

        print(f"Max d3body: {torch.max(descriptors_3body)}")

        # concatenate radial descriptors and 3body descriptors

        descriptors = torch.cat([rbf, descriptors_3body], dim=1) # num_pairs x num_descriptors
        assert(descriptors.size()[0] == rij.size()[0])

        # input descriptors to a network for each pair; calculate pairwise energies

        print(rbf.size())
        print(descriptors_3body.size())
        print(descriptors.size())

        eij = self.model(descriptors)

        print(f"Max eij: {torch.max(eij)}")

        #for param_tensor in self.state_dict():
        #    print(param_tensor, "\t", self.state_dict()[param_tensor]) 

        assert(cutoff_functions.size() == eij.size())
        eij = torch.mul(eij,cutoff_functions)
        assert(eij.size()[0] == rij.size()[0])

        # differentiate energy wrt interatomic displacements (rij)
        energy = torch.sum(eij)
        #print(energy/54.)
        gradients_wrt_rij = torch.autograd.grad(energy, rij)[0]
        beta[:,:] = gradients_wrt_rij.detach().cpu().numpy().astype(np.float64)

        # differentiate pairwise energies wrt rij
        # NOTE: this doesn't work, gives (npair,3) derivative tensor but should be more like a Jacobian
        #depair_drij = torch.autograd.grad(eij, 
        #                        rij, 
        #                        grad_outputs=torch.ones_like(eij))[0]

        #print("Size of gradients_wrt_rij:")
        #print(gradients_wrt_rij.size())

        #print("Size of depair_drij:")
        #print(depair_drij.size())

        # calculate force on atom 1
        # this gives correct force on atom 1 (tag-1 = 0), compare with force_comparison.dat:
        """
        npairs = rij.size()[0]
        f0x = 0.
        f0y = 0.
        f0z = 0.
        for p in range(0,npairs):
            if tag_i[p] == 0:
                f0x -= gradients_wrt_rij[p,0]
                f0y -= gradients_wrt_rij[p,1]
                f0z -= gradients_wrt_rij[p,2]
            elif tag_j[p] == 0:
                f0x += gradients_wrt_rij[p,0]
                f0y += gradients_wrt_rij[p,1]
                f0z += gradients_wrt_rij[p,2]
        print(-1.0*f0x)
        print(-1.0*f0y)
        print(-1.0*f0z)
        #print(tag_i.size())
        """

        # this also gives the correct force for atom 1 (tag-1 = 0) using a mask:
        """
        mask_i = tag_i == 0
        mask_j = tag_j == 0
        add = torch.sum(gradients_wrt_rij[mask_i], axis=0)
        subtract = torch.sum(gradients_wrt_rij[mask_j], axis=0)
        print(add-subtract)
        """

        # might be best to give this function a pointer to dE/dRij which gets
        # populated inside here, then do the loop to calculate per-atom forces
        # in LAMMPS later.

        # useful to see how autograd works:
        """
        energy = 2.0*rij
        energy = torch.sum(energy)
        #gradients_wrt_rij = torch.autograd.grad(energy, 
        #                        rij, 
        #                        grad_outputs=torch.ones_like(energy))[0]
        # this also works and looks simpler:
        gradients_wrt_rij = torch.autograd.grad(energy, rij)[0]
        print(gradients_wrt_rij)
        """

        """
        descriptors = torch.from_numpy(descriptors).to(dtype=self.dtype, device=self.device).requires_grad_(True)
        elems = torch.from_numpy(elems).to(dtype=torch.long, device=self.device) - 1

        with torch.autograd.enable_grad():

            energy_nn = self.model(descriptors, elems)
            if energy_nn.ndim > 1:
                energy_nn = energy_nn.flatten()

            beta_nn = torch.autograd.grad(energy_nn.sum(), descriptors)[0]

        beta[:] = beta_nn.detach().cpu().numpy().astype(np.float64)
        energy[:] = energy_nn.detach().cpu().numpy().astype(np.float64)
        """