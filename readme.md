## About GTasb3D

GTasb3D is a 3D thermophysical code using the General finite difference model to simulate the Thermal evolution of active small bodies. The features of this code includes:
* Capable of simulating various thermophysical processes involved in the transfer of heat fluxes, the sublimation/condensation of ice, and the diffusion of gas through a porous matrix within an icy body.
* Applicable to modeling a body with various kinds of shapes.
* Parallelized implimentation in C++ with reasonable accuracy and efficiency.
More details on the mathematical formulate, code architecture, and validation tests can be found in <a href="https://doi.org/10.3847/PSJ/acc4c4">Zhang \& Hartzell (2023)</a>

## About this repository

Here includes the GTasb3D source code as well as an example with input files and MATLAB scripts. 

To compile the code, create a folder, e.g., "bin", in the top directory, type
```sh
cmake ../
```
This will generate a Makefile in the folder, and type
```sh
make
```
Voilà!  You should find the executable file named "GTasb3D" in this folder, and now you can start to perform a GTasb3D simulation.

Steps to run Distributed GTasb3D:
* Copy the obtained GTasb3D to the simulation directory, e.g., example/133P.
* Edit "gfdm.dat" file to adjust the simulation setup and physical parameters.
* Create a folder named "result" in this directory and launch a simulation by 
```sh
mpirun -n $NUMBER_OF_PROCESSORS ../../bin/GTasb3D  gfdm.dat
```

Please cite <a href="https://doi.org/10.3847/PSJ/acc4c4">Zhang \& Hartzell (2023)</a> if you use this code for your study. Please contact the author via the gmail address at yzhangastro if you would like to have more details about how to use and modify this code. 


## Distributed version

MPICH and p4est integrated to parallelize computation in a distributed compute fashion. 

include/split.hpp encapsulates all additional data structures and functions added to original code.

A 3D p8est mesh was used, with customizable number of octants using p4est_box_size in include/split.hpp, 
resulting in a mesh sizing of [0, p4est_box_size]^3, which would get assigned to ranks evenly using p4est's tree-based locality.

Respective per-node neighbors are found and statically tracked before data computation, transferring Temperature and Conductivity values at each timestep to other ranks. This data is serialized and transferred via MPICH.

Global node identifiers are maintained.

Matrix multiplication is left untouched, but all neighbor information is made present at the time of local calculation. 

All OpenMP code removed. 

Output node information is per rank, delivered to the same output file 
-> gdfmrankX.timestep e.g gdfmrank0.0000000000

## To be worked on

include/common.hpp defines USE_SELF and USE_GAS for self-shadow, self-heating, gas, and ice activity.
This implementation does not touch on these, and requires these definitions to be turned off.

Testing for even octant distribution across ranks, even node count distribution across ranks, and present neighbor information after MPICH transfer all work and seem to be in order. 

Final rank-combined output diff testing to the original implementation is yet to be completed.

Code style can be cleaned up, and iostream can likely be removed as well.



