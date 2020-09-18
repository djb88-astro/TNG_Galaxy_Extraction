import os
import h5py
import numpy as np

"""
This class reads data from the TNG snapshot files. Given a
halo number (FoF group ID), it then extracts all particles
in the group for the datasets of interest
"""

class halo:
    def __init__(self, mpi, subfind_table, j):
        # Store key halo properties for convenience
        self.hub = subfind_table.hub
        self.axp = subfind_table.axp
        self.redshift = subfind_table.redshift
        self.CoP = subfind_table.CoP[j]
        self.M200 = subfind_table.M200[j]
        self.R200 = subfind_table.R200[j]
        self.Vbulk = subfind_table.Vbulk[j]

        # Compute group length and offset
        offsets = np.zeros(subfind_table.GrLenType.shape, dtype=np.int)
        offsets[1:] = np.cumsum(subfind_table.GrLenType[:-1], axis=0)
        self.LenType = subfind_table.GrLenType[j]
        self.OffType = offsets[j]
        return

    def read_datasets(
            self,
            mpi,
            doi,
            sim='/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output',
            snap=99
    ):
        """
        Read all datasets of interest.

        Arguments:
          -mpi  : An instance of the MPI class
          -doi  : Datasets to read [LIST]
          -sim  : Path to the simulation to be read [STRING]
          -snap : Snapshot to be read [INT]
        """

        self.which_snapdir_files(sim, snap, 0)

        f = h5py.File(self.files[0], 'r')
        self.U_mass = f['Header'].attrs['UnitMass_in_g']
        self.U_length = f['Header'].attrs['UnitLength_in_cm']
        self.U_velcty = f['Header'].attrs['UnitVelocity_in_cm_per_s']
        self.BoxSize = f['Header'].attrs['BoxSize'] * self.U_length * self.axp / self.hub
        self.OmegaB = f['Header'].attrs['OmegaBaryon']
        self.OmegaM = f['Header'].attrs['Omega0']
        self.OmegaL = f['Header'].attrs['OmegaLambda']
        f.close()
        self.ufrac = self.OmegaB / self.OmegaM

        for x in doi:
            ptype = int(x[8])
            if ptype != self.ptype:
                self.which_snapdir_files(sim, snap, ptype)
            self.read_dataset(mpi, x)
        return

    def which_snapdir_files(self, sim, snap, ptype):
        """
        Gather all files in a snapshot directory, examine offsets relative to
        number of a given PartType in the files and return relevant one

        Arguments:
          -sim   : Path to the simulation to be read [STRING]
          -snap  : Snapshot to be read [INT]
          -ptype : Particle type of interest [INT]
        """

        files = []
        for x in os.listdir('{0}/snapdir_{1:03d}/'.format(sim, snap)):
            if x.startswith('snap_'):
                files.append('{0}/snapdir_{1:03d}/{2}'.format(sim, snap, x))

        if len(files) > 1:
            sort_order = np.argsort(np.array([x.split('.', 2)[1] for x in files], dtype=np.int))
            files = list(np.array(files)[sort_order])
            del sort_order

        offset     = 0
        rqrd_files = []
        chunk      = []
        for x in files:
            f = h5py.File(x, 'r')
            npart = f['Header'].attrs['NumPart_ThisFile'][ptype]
            f.close()

            if offset + npart < self.OffType[ptype]:
                offset += npart
                continue

            rqrd_files.append(x)
            if offset <= self.OffType[ptype]:
                start = self.OffType[ptype] - offset
            else:
                start = 0

            if offset + npart <= self.OffType[ptype] + self.LenType[ptype]:
                finish = npart
                chunk.append([start, finish])
            else:
                finish = start + (self.OffType[ptype] + self.LenType[ptype]) - (offset + start)
                chunk.append([start, finish])
                break
            offset += npart
        del files, offset
        chunk = np.array(chunk)

        self.ptype = ptype
        self.files = rqrd_files
        self.chunk = chunk
        return

    def read_dataset(self, mpi, x):
        """
        Read and store chunks of the HDF5 files based on the calculated offsets

        Arguments:
          -mpi : An instance of the MPI class
          -x   : Dataset to be read [STRING]
        """

        if mpi.Rank == 0:
            print('  -{0}'.format(x), flush=True)

        npart = np.sum(self.chunk[:,1] - self.chunk[:,0])

        if x != 'PartType1/Masses':
            f = h5py.File(self.files[0], 'r')
            dtype = f[x].dtype
            shape = f[x].shape
            a_scl = f[x].attrs['a_scaling']
            h_scl = f[x].attrs['h_scaling']
            u2cgs = f[x].attrs['to_cgs']
            f.close()

            if len(shape) > 1:
                npart = (npart, shape[1])
                dset  = np.zeros(npart, dtype=dtype)
            else:
                dset = np.zeros(npart, dtype=dtype)
            del npart, dtype, shape

            offset = 0
            for j in range(0, len(self.files), 1):
                f = h5py.File(self.files[j], 'r')
                start = self.chunk[j,0]
                finish = self.chunk[j,1]
                dset[offset:offset+(finish-start)] = f[x][start:finish]
                f.close()
                offset += finish - start
            del offset
        else:
            f = h5py.File(self.files[0], 'r')
            DMmass = f['Header'].attrs['MassTable'][1]
            f.close()
            u2cgs = self.U_mass
            h_scl = -1
            dset = np.zeros(npart, dtype=np.float64) + DMmass

        if x == 'PartType0/Coordinates':
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5*dims, pos-dims, pos)
            pos = np.where(pos < -0.5*dims, pos+dims, pos)
            self.pos = pos
            self.rad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == 'PartType0/Density':
            self.rho = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
        elif x == 'PartType0/ElectronAbundance':
            self.ne_nh = dset
        elif x == 'PartType0/InternalEnergy':
            self.inte = dset*u2cgs
        elif x == 'PartType0/Masses':
            self.mass = dset * u2cgs * (self.hub ** h_scl)
        elif x == 'PartType0/GFM_CoolingRate':
            self.Crate = dset
        elif x == 'PartType0/GFM_Metals':
            self.metals = dset
        elif x =='PartType0/GFM_Metallicity':
            self.Ztot = dset / 0.0127 # primordial solar (apparently)
        elif x == 'PartType0/StarFormationRate':
            self.SFR = dset
        elif x == 'PartType0/Velocities':
            self.vel = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
        elif x == 'PartType1/Coordinates':
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5 * dims, pos - dims, pos)
            pos = np.where(pos < -0.5 * dims, pos + dims, pos)
            self.DMpos = pos
            self.DMrad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == 'PartType1/Masses':
            self.DMmass = dset * u2cgs * (self.hub ** h_scl)
        elif x == 'PartType4/Coordinates':
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5 * dims, pos - dims, pos)
            pos = np.where(pos < -0.5 * dims, pos + dims, pos)
            self.STpos = pos
            self.STrad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == 'PartType4/Masses':
            self.STmass  = dset * u2cgs * (self.hub ** h_scl)
        elif x == 'PartType4/GFM_StellarFormationTime':
            self.STsft = dset
        else:
            if mpi.Rank == 0:
                print(' DATASET STORE NOT SET FOR {0}!!!'.format(x), flush=True)
            quit()
        return

    def compute_morphological_metrics(self):
        """
        Compute the theoretical morphological metrics of this halo
        """

        print(" ")
        print(sorted(self.__dict__.keys()))
        quit()
