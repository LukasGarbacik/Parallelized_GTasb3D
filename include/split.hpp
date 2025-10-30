#include "fdmnode.hpp"
//C linked headers for current imp to be able to use
extern "C" {
    #include <p8est_bits.h>
    #include <p8est_extended.h>
    #include <p8est_connectivity.h>
    #include <p8est_mesh.h>
    #include <p8est_ghost.h>
    #include <p8est_geometry.h>
    #include <sc.h>
    #include <mpi.h>
}

#define p4est_box_size 100

typedef struct {
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;
} octant_bounds_t;

typedef struct {
    octant_bounds_t bounds;
    int octant_id;
    int mpirank;
} octant_data_t;

typedef struct {
    sc_MPI_Comm         mpicomm;
    int                 mpirank;
    int                 mpisize;
} mpi_context_t;

typedef struct{
    mpi_context_t * mpi; 
    p8est_connectivity_t * connectivity;
    p8est_t * p8est; 
    p8est_ghost_t *ghost;
    p8est_mesh_t * mesh;

    int num_octants;
} local_data_t;

typedef struct {
    std::vector<fdmnode> NodeVec;
} sub_master_t;


void mpi_set(int argc, char **argv, local_data_t *data);
void p8est_setup(local_data_t *data);
void cleanup(local_data_t * g);
void oct_init(p8est_t *p8est, p4est_topidx_t which_tree, p8est_quadrant_t *octant);

void populate_oct_bounds(local_data_t * g, p4est_topidx_t which_tree, p8est_quadrant_t *octant);

void scale_particles(Master * S, local_data_t * data, double bound[6]);
void send_master_info(Master * S, local_data_t * data);

//Lukas implementation:

//One master per rank,
//rank 0 gets all inital file information using Master::initSystem
//broadcast happens populating the S.