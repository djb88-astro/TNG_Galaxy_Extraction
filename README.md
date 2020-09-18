# TNG_Galaxy_Extraction
This code reconstructs the Subfind table at a given snapshot of a simulation, selects haloes above a given mass and then extracts resolved galaxies and compute morpholigcal metrics.
Resolved is defined as 100x the dark matter particle mass of the simulation.
Additionally, its then checks to ensure that there is a stellar mass of 50x the target mass gass within twice the stellar half mass radius so ensure the stellar component of any subhalo is sufficiently well sampled.
Both of these cuts can chosen depending on user preference.
All morphological metrics [Energy ratio (Barnes+ 2017b), Fsub (Neto+ 2007), Xoff (Thomas+ 2001)] are computed with R200.
