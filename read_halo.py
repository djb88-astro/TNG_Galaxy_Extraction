import os
import h5py
import numpy as np
import constants as ct

"""
This class reads data from the TNG snapshot files. Given a
halo number (FoF group ID), it then extracts all particles
in the group for the datasets of interest
"""


class halo:
    def __init__(self, mpi, sub_tab, j):
        # Store key halo properties for convenience - units to cgs
        self.hub = sub_tab.hub
        self.axp = sub_tab.axp
        self.redshift = sub_tab.redshift
        self.CoP = sub_tab.CoP[j] * ct.kpc_cm * sub_tab.axp / sub_tab.hub
        self.M200 = sub_tab.M200[j] * ct.Mtng_Msun * ct.Msun_g / sub_tab.hub
        self.R200 = sub_tab.R200[j] * ct.kpc_cm * sub_tab.axp / sub_tab.hub
        self.Vbulk = sub_tab.Vbulk[j] * ct.km_cm / self.axp

        # Compute group length and offset
        offsets = np.zeros(sub_tab.GrLenType.shape, dtype=np.int)
        offsets[1:] = np.cumsum(sub_tab.GrLenType[:-1], axis=0)
        self.LenType = sub_tab.GrLenType[j]
        self.OffType = offsets[j]

        # Compute subhalo length
        tmp = sub_tab.SubLenType[
            sub_tab.FirstSub[j] : sub_tab.FirstSub[j] + sub_tab.Nsubs[j]
        ]
        self.SubLenType = np.zeros(tmp.shape, dtype=np.int)
        self.SubLenType[1:] = np.cumsum(tmp[:-1], axis=0)
        return

    def read_datasets(
        self,
        mpi,
        doi,
        sim="/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output",
        snap=99,
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

        f = h5py.File(self.files[0], "r")
        self.U_mass = f["Header"].attrs["UnitMass_in_g"]
        self.U_length = f["Header"].attrs["UnitLength_in_cm"]
        self.U_velcty = f["Header"].attrs["UnitVelocity_in_cm_per_s"]
        self.BoxSize = (
            f["Header"].attrs["BoxSize"] * self.U_length * self.axp / self.hub
        )
        self.OmegaB = f["Header"].attrs["OmegaBaryon"]
        self.OmegaM = f["Header"].attrs["Omega0"]
        self.OmegaL = f["Header"].attrs["OmegaLambda"]
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
        for x in os.listdir("{0}/snapdir_{1:03d}/".format(sim, snap)):
            if x.startswith("snap_"):
                files.append("{0}/snapdir_{1:03d}/{2}".format(sim, snap, x))

        if len(files) > 1:
            sort_order = np.argsort(
                np.array([x.split(".", 2)[1] for x in files], dtype=np.int)
            )
            files = list(np.array(files)[sort_order])
            del sort_order

        offset = 0
        rqrd_files = []
        chunk = []
        for x in files:
            f = h5py.File(x, "r")
            npart = f["Header"].attrs["NumPart_ThisFile"][ptype]
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
                finish = (
                    start
                    + (self.OffType[ptype] + self.LenType[ptype])
                    - (offset + start)
                )
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
            print("  -{0}".format(x), flush=True)

        npart = np.sum(self.chunk[:, 1] - self.chunk[:, 0])

        if x != "PartType1/Masses":
            f = h5py.File(self.files[0], "r")
            dtype = f[x].dtype
            shape = f[x].shape
            a_scl = f[x].attrs["a_scaling"]
            h_scl = f[x].attrs["h_scaling"]
            u2cgs = f[x].attrs["to_cgs"]
            f.close()

            if len(shape) > 1:
                npart = (npart, shape[1])
                dset = np.zeros(npart, dtype=dtype)
            else:
                dset = np.zeros(npart, dtype=dtype)
            del npart, dtype, shape

            offset = 0
            for j in range(0, len(self.files), 1):
                f = h5py.File(self.files[j], "r")
                start = self.chunk[j, 0]
                finish = self.chunk[j, 1]
                dset[offset : offset + (finish - start)] = f[x][start:finish]
                f.close()
                offset += finish - start
            del offset
        else:
            f = h5py.File(self.files[0], "r")
            DMmass = f["Header"].attrs["MassTable"][1]
            f.close()
            u2cgs = self.U_mass
            h_scl = -1
            dset = np.zeros(npart, dtype=np.float64) + DMmass

        if x == "PartType0/Coordinates":
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5 * dims, pos - dims, pos)
            pos = np.where(pos < -0.5 * dims, pos + dims, pos)
            self.pos = pos
            self.rad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == "PartType0/Density":
            self.rho = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
        elif x == "PartType0/ElectronAbundance":
            self.ne_nh = dset
        elif x == "PartType0/InternalEnergy":
            self.inte = dset * u2cgs
        elif x == "PartType0/Masses":
            self.mass = dset * u2cgs * (self.hub ** h_scl)
        elif x == "PartType0/GFM_CoolingRate":
            self.Crate = dset
        elif x == "PartType0/GFM_Metals":
            self.metals = dset
        elif x == "PartType0/GFM_Metallicity":
            self.Ztot = dset / 0.0127  # primordial solar (apparently)
        elif x == "PartType0/StarFormationRate":
            self.SFR = dset
        elif x == "PartType0/Velocities":
            self.vel = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
        elif x == "PartType1/Coordinates":
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5 * dims, pos - dims, pos)
            pos = np.where(pos < -0.5 * dims, pos + dims, pos)
            self.DMpos = pos
            self.DMrad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == "PartType1/Masses":
            self.DMmass = dset * u2cgs * (self.hub ** h_scl)
        elif x == "PartType4/Coordinates":
            pos = dset * u2cgs * (self.axp ** a_scl) * (self.hub ** h_scl)
            pos -= self.CoP
            dims = np.array([self.BoxSize, self.BoxSize, self.BoxSize])
            pos = np.where(pos > 0.5 * dims, pos - dims, pos)
            pos = np.where(pos < -0.5 * dims, pos + dims, pos)
            self.STpos = pos
            self.STrad = np.sqrt((pos ** 2.0).sum(axis=-1))
            del pos
        elif x == "PartType4/Masses":
            self.STmass = dset * u2cgs * (self.hub ** h_scl)
        elif x == "PartType4/GFM_StellarFormationTime":
            self.STsft = dset
        else:
            if mpi.Rank == 0:
                print(" DATASET STORE NOT SET FOR {0}!!!".format(x), flush=True)
            quit()
        return

    def compute_gas_temperature(self):
        """
        Compute the temperature of every gas cell
        """

        mu = (4.0 * ct.mp_g) / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * self.ne_nh)
        self.temp = (2.0 / 3.0) * (self.inte / ct.kB_erg_K) * mu
        return

    def compute_morphological_metrics(self):
        """
        Compute the theoretical morphological metrics of this halo
        """

        keys = self.__dict__.keys()

        # Centre of mass offset
        self.Xoff = 0.0
        weights = 0.0

        if "mass" in keys:
            gdx = np.where(self.rad <= self.R200)[0]
            self.Xoff += np.sum(self.mass[gdx] * self.pos[gdx].T, axis=-1)
            weights += np.sum(self.mass[gdx])

        if "DMmass" in keys:
            ddx = np.where(self.DMrad <= self.R200)[0]
            self.Xoff += np.sum(self.DMmass[ddx] * self.DMpos[ddx].T, axis=-1)
            weights += np.sum(self.DMmass[ddx])

        if "STmass" in keys:
            sdx = np.where(self.STrad <= self.R200)[0]
            self.Xoff += np.sum(self.STmass[sdx] * self.STpos[sdx].T, axis=-1)
            weights += np.sum(self.STmass[sdx])

        if weights > 0.0:
            self.Xoff /= weights
            self.Xoff = np.sqrt((self.Xoff ** 2.0).sum(axis=-1)) / self.R200

        # Substructure fraction
        Mtot = (
            np.sum(self.mass[gdx]) + np.sum(self.DMmass[ddx]) + np.sum(self.STmass[sdx])
        )

        Msub = 0.0
        if "mass" in keys:
            idx = (
                np.arange(self.SubLenType[-1, 0] - self.SubLenType[1, 0] + 1)
                + self.SubLenType[1, 0]
            )
            idx = np.intersect1d(gdx, idx, assume_unique=True)
            Msub += self.mass[idx].sum()

        if "DMmass" in keys:
            idx = (
                np.arange(self.SubLenType[-1, 1] - self.SubLenType[1, 1] + 1)
                + self.SubLenType[1, 1]
            )
            idx = np.intersect1d(ddx, idx, assume_unique=True)
            Msub += self.DMmass[idx].sum()

        if "STmass" in keys:
            idx = (
                np.arange(self.SubLenType[-1, 4] - self.SubLenType[1, 4] + 1)
                + self.SubLenType[1, 4]
            )
            idx = np.intersect1d(sdx, idx, assume_unique=True)
            Msub += self.STmass[idx].sum()

        if Mtot > 0.0:
            self.Fsub = Msub / Mtot

        # Kinetic ratio
        self.Erat = 0.0
        if "mass" in keys:
            Ekin = np.sum(
                0.5
                * self.mass[gdx]
                * ((self.vel[gdx] - self.Vbulk) ** 2.0).sum(axis=-1)
            )
            Ethm = np.sum(
                1.5 * self.mass[gdx] / (ct.mu * ct.mp_g) * ct.kB_erg_K * self.temp[gdx]
            )
            self.Erat = Ekin / Ethm

        return
