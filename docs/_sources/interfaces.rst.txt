Interfaces Module
-----------------

This module contains the functions needed to interface ML_Calc_Driver with other 
python packages. For the moment, these packages include `ase <https://wiki.fysik.dtu.dk/ase/index.html>`_
and `SchnetPack <https://github.com/atomistic-machine-learning/schnetpack>`_. More
interfaces might be needed when adding new calculators.

.. automodule:: mlcalcdriver.interfaces

.. toctree::
   :maxdepth: 1
   :caption: Interfaces

   ase_interface
   schnet_interface
   atoms_to_patches
