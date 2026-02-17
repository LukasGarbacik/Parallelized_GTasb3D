// Guard to prevent multiple definitions
#ifndef split_hpp
#define split_hpp

#include "common.hpp"
#include <map>

// Forward declarations to avoid circular includes
class fdmnode;
class Master;

// C linked headers for current imp to be able to use
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

#define p4est_box_size 2

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

typedef struct {
    unsigned int nid;
    double * temp;
    double * cond;
} temp_mpi_data_t;

typedef struct {
    unsigned int nid;
    double temp;
    double cond;
} byte_node_t;

typedef struct{
    mpi_context_t * mpi; 
    p8est_connectivity_t * connectivity;
    p8est_t * p8est; 
    p8est_ghost_t *ghost;
    p8est_mesh_t * mesh;

    std::map<unsigned int, fdmnode *> local_nodes;
    std::map<unsigned int, temp_mpi_data_t> global_data;
    std::vector<unsigned int> needed;
    std::vector<std::vector<temp_mpi_data_t>> outgoing;

    std::vector<unsigned int> incoming_counts;
    std::vector<int> offsets;
    std::vector<byte_node_t> recv_buffer;

    int num_octants;
} local_data_t;

//set extern so master can use this object
extern local_data_t *g_data;


void mpi_set(int argc, char **argv, local_data_t *data);
void p8est_setup(local_data_t *data);
void cleanup(local_data_t * g);
void oct_init(p8est_t *p8est, p4est_topidx_t which_tree, p8est_quadrant_t *octant);

bool in_bounds(vector3d scaled_pos, octant_bounds_t bounds);
void populate_oct_bounds(local_data_t * g, p4est_topidx_t which_tree, p8est_quadrant_t *octant);

void scale_particles(Master * S, local_data_t * data, double bound[6]);
void gather_bounaries(local_data_t * data);
void send_master_info(Master * S, local_data_t * data);
void mpi_send_lookup_table(local_data_t * data);
void mpi_recv_lookup_table(local_data_t * data);
void mpi_copy_rank0_lookup_to_init_data(local_data_t * data);
void populate_inital_outgoing(Master * S, local_data_t * data);
void populate_lookup_table(Master * S, local_data_t * data);

void print_bounds(int rank, octant_bounds_t *bounds);
void print_particle(int rank, int bound_num, fdmnode * particle);
void print_lookup_tables(local_data_t * data);
void print_local_lookup_table(local_data_t * data);

//Lukas implementation:

//One master per rank,
//rank 0 gets all inital file information using Master::initSystem
//broadcast happens populating the S.

#endif // split_hpp