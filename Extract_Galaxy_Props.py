import h5py
import mpi_init
import subfind_data
import read_halo
import merge
import constants as ct
import numpy as np


def extract_galaxy_colours(
    mpi,
    sim,
    path,
    bar_mass,
    dm_mass,
    snapshot,
    m_thres=1.0e14,
    f_resolve=100.0,
    extent=2.0,
):
    """
    Extract the colours of all galaxies within a 2 * R200 sphere around
    all clusters with a mass M200 > 10^14 Msun

    Arguments:
      -mpi       : MPI environment class instance
      -sim       : Tag labelling the simulation of interest [STRING]
      -path      : Path the simulation data [STRING]
      -bar_mass  : Target mass of gas cells [FLOAT]
      -dm_mass   : Mass of dark matter particles [FLOAT]
      -snapshot  : The snapshot of interest [INT]
      -m_thres   : Mass threshold for halo selection in solar masses [FLOAT]
      -extent    : Extraction radius factor [FLOAT]
      -f_resolve : Multiplicative factor determining if a subhalo is sufficiently resolved [FLOAT]
    """

    if not mpi.Rank:
        print("--- Examining {0} simulation".format(sim), flush=True)
        print("--- Snapshot: {0:03d}".format(snapshot), flush=True)

    # Load subfind table
    quantities_of_interest = [
        "Group/GroupPos",
        "Group/Group_M_Crit200",
        "Group/Group_R_Crit200",
        "Group/GroupLenType",
        "Group/GroupFirstSub",
        "Group/GroupNsubs",
        "Group/GroupVel",
        "Subhalo/SubhaloLenType",
        "Subhalo/SubhaloMassInRadType",
        "Subhalo/SubhaloMassType",
        "Subhalo/SubhaloPos",
        "Subhalo/SubhaloStellarPhotometrics",
    ]
    subfind_table = subfind_data.build_table(
        mpi, quantities_of_interest, sim=path, snap=snapshot
    )

    # Indentify haloes above mass threshold
    hdx = np.where(subfind_table.M200 * ct.Mtng_Msun / subfind_table.hub >= m_thres)[0]

    # Indentify resolved subhalos
    rdx = np.where(
        subfind_table.SubMassType[:, 1] * ct.Mtng_Msun / subfind_table.hub
        >= f_resolve * dm_mass
    )[0]

    # Indentify "galaxies" -- i.e. subhalos with sufficient stellar mass
    sdx = np.where(
        subfind_table.SubMassRadType[:, 4] * ct.Mtng_Msun / subfind_table.hub
        >= f_resolve * bar_mass
    )[0]

    # Now combine to find resolved galaxies in this snapshot
    gdx = np.intersect1d(rdx, sdx, assume_unique=True)

    # Loop over selected haloes
    for j in range(mpi.Rank, hdx.size, mpi.NProcs):
        if not mpi.Rank:
            print(" > Halo {0:04d}".format(hdx[j]), flush=True)

        # Extract all subhaloes within extent * R200
        Spos = subfind_table.SubPos[gdx] - subfind_table.CoP[hdx[j]]
        Srad = np.sqrt((Spos ** 2.0).sum(axis=-1))
        Sphot = subfind_table.SubPhoto[gdx]

        idx = np.where(Srad <= extent * subfind_table.R200[hdx[j]])[0]

        # Now load particles with FoF group for morphological metrics
        halo = read_halo.halo(mpi, subfind_table, hdx[j])

        datasets_of_interest = [
            "PartType0/Coordinates",
            "PartType0/ElectronAbundance",
            "PartType0/InternalEnergy",
            "PartType0/Masses",
            "PartType0/Velocities",
            "PartType1/Coordinates",
            "PartType1/Masses",
            "PartType4/Coordinates",
            "PartType4/Masses",
        ]
        halo.read_datasets(mpi, datasets_of_interest, sim=path, snap=snapshot)

        if not mpi.Rank:
            print("  > Computing gas temperature", flush=True)
        halo.compute_gas_temperature()

        if not mpi.Rank:
            print("  > Computing theoretical morphological metrics", flush=True)
        halo.compute_morphological_metrics()

        # Save halo and subhalo properties
        f = h5py.File(
            "output/{0}_z{1:d}p{2:02d}_{3:03d}.hdf5".format(
                sim,
                int(subfind_table.redshift),
                int(100.0 * (subfind_table.redshift - int(subfind_table.redshift))),
                mpi.Rank,
            ),
            "a",
        )
        grp = f.create_group("halo_{0:04d}".format(hdx[j]))
        grp.attrs["M200_Msun"] = (
            subfind_table.M200[hdx[j]] * ct.Mtng_Msun / subfind_table.hub
        )
        grp.attrs["R200_kpc"] = (
            subfind_table.R200[hdx[j]] * subfind_table.axp / subfind_table.hub
        )
        grp.attrs["Eratio"] = halo.Erat
        grp.attrs["Fsub"] = halo.Fsub
        grp.attrs["Xoff"] = halo.Xoff
        f.create_dataset(
            "halo_{0:04d}/Positions_kpc".format(hdx[j]),
            data=Spos[idx] * subfind_table.axp / subfind_table.hub,
        )
        f.create_dataset(
            "halo_{0:04d}/Magnitudes_UBVKgriz".format(hdx[j]), data=Sphot[idx]
        )
        f.close()

    # Merge output files
    mpi.comm.Barrier()
    if not mpi.Rank:
        merge.merge_outputs(
            "{0}_z{1:d}p{2:02d}".format(
                sim,
                int(subfind_table.redshift),
                int(100.0 * (subfind_table.redshift - int(subfind_table.redshift))),
            )
        )
    mpi.comm.Barrier()
    return


if __name__ == "__main__":

    # Set up simulation dictionary - path, target baryonic mass, dark matter mass (no h)
    sims = {
        "TNG300-L3": [
            "/n/hernquistfs3/IllustrisTNG/Runs/L205n625TNG/output",
            7.0e8,
            3.8e9,
        ],
        "TNG300-L2": [
            "/n/hernquistfs3/IllustrisTNG/Runs/L205n1250TNG/output",
            8.8e7,
            4.7e8,
        ],
        "TNG300-L1": [
            "/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output",
            1.1e7,
            5.9e7,
        ],
    }

    # Snapshots of interest
    snapshots = [40, 50, 59, 67, 72, 78, 84, 91, 99]

    # Initialize MPI environment
    mpi = mpi_init.mpi()

    # Now loop and extract galaxy colours
    for res, props in sims.items():
        for snap in snapshots:
            extract_galaxy_colours(mpi, res, props[0], props[1], props[2], snap)
