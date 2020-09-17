import h5py
import subfind_data
import mpi_init
import merge


def extract_galaxy_colours(sim, path, bar_mass, dm_mass, snapshot):
    """
    Extract the colours of all galaxies within a 2 * R200 sphere around
    all clusters with a mass M200 > 10^14 Msun

    Arguments:
      -sim      : Tag labelling the simulation of interest [STRING]
      -path     : Path the simulation data [STRING]
      -bar_mass : Target mass of gas cells [FLOAT]
      -dm_mass  : Mass of dark matter particles [FLOAT]
      -snapshot : The snapshot of interest [INT]
    """

    # Load subfind table


    # Copy


    # Only select halos above mass threshold


    # Loop over selected haloes


    # Extract all subhaloes within 2 * R200


    # Save halo


    # Merge output files



if __name__ == "__main__":

    # Set up simulation dictionary - path, target baryonic mass, dark matter mass (no h)
    sims = {
        "TNG300_L3": ["/n/hernquistfs3/IllustrisTNG/Runs/L205n625TNG/output", 7.0e8, 3.8e9],
        "TNG300_L2": ["/n/hernquistfs3/IllustrisTNG/Runs/L205n1250TNG/output", 8.8e7, 4.7e8],
        "TNG300_L1": ["/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output", 1.1e7, 5.9e7]
    }

    # Snapshots of interest
    snapshots = [40, 50, 59, 67, 72, 78, 84, 91, 99]

    # Now loop and extract galaxy colours
    for res, props in sims.items():
        for snap in snapshots:
            extract_galaxy_colours(res, props[0], props[1], props[2], snap)
            quit()
