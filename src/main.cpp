/*****************************************************************************************************************************
 *
 *  GTasb3D:  C++ program for generalized finite differences thermal modeling
 *
 *  Developed by Yun Zhang (2022-2023)
 *
 *  Function: solve 3D transient heat conduction in a given irregular-shape body
 *  Feature:
 *
 *
 *****************************************************************************************************************************/


#include "master.hpp"
#include "split.hpp"
#include <ctime>
#include <time.h>


void cleanup(local_data_t * g){
    if(!g) return;

    if(g->mesh) {
        p8est_mesh_destroy(g->mesh);
        g->mesh = NULL;
    }
    if(g->ghost) {
        p8est_ghost_destroy(g->ghost);
        g->ghost = NULL;
    }
    if(g->p8est) {
        p8est_destroy(g->p8est);
        g->p8est = NULL;
    }
    if(g->connectivity) {
        p8est_connectivity_destroy(g->connectivity);
        g->connectivity = NULL;
    }
    
    if(g->mpi) {
        sc_MPI_Barrier(g->mpi->mpicomm);
        int mpiret = sc_MPI_Finalize();
        SC_CHECK_MPI(mpiret);
    }
}


void mpi_set(int argc, char **argv, local_data_t * data) {
    if(!data || !data->mpi) return;
    
    int mpiret = sc_MPI_Init(&argc, &argv);
    SC_CHECK_MPI(mpiret);

    data->mpi->mpicomm = sc_MPI_COMM_WORLD;
    sc_MPI_Comm_rank(data->mpi->mpicomm, &data->mpi->mpirank);
    sc_MPI_Comm_size(data->mpi->mpicomm, &data->mpi->mpisize);
    
    sc_init(data->mpi->mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
}

void populate_oct_bounds(local_data_t * g, p4est_topidx_t which_tree, p8est_quadrant_t *octant){
    if(!g || !octant || !octant->p.user_data) return;
    
    octant_data_t *data = (octant_data_t *) octant->p.user_data;

    int x = octant->x;
    int y = octant->y;
    int z = octant->z;

    double vertex[3];
    p8est_qcoord_to_vertex(g->connectivity, which_tree, x, y, z, vertex);

    data->bounds.x_min = vertex[0];
    data->bounds.x_max = vertex[0] + 1.0;
    data->bounds.y_min = vertex[1];
    data->bounds.y_max = vertex[1] + 1.0;
    data->bounds.z_min = vertex[2];
}
/*
bool in_bounds(vector3d * scaled_pos, octant_bounds_t * bounds){
    if(!scaled_pos || !bounds) return false;
    
    return (scaled_pos->x >= bounds->x_min && scaled_pos->x <= bounds->x_max &&
            scaled_pos->y >= bounds->y_min && scaled_pos->y <= bounds->y_max &&
            scaled_pos->z >= bounds->z_min && scaled_pos->z <= bounds->z_max);
}
*/
void oct_init(p8est_t *p8est, p4est_topidx_t which_tree, p8est_quadrant_t *octant){
    if(!p8est || !octant) return;
    
    local_data_t *data = (local_data_t *) p8est->user_pointer;
    if(!data) return;
    
    octant->p.user_data = calloc(1, sizeof(octant_data_t));
    if(!octant->p.user_data) {
        printf("ERROR: Failed to allocate octant data\n");
        return;
    }
    
    octant_data_t *oct = (octant_data_t *) octant->p.user_data;

    populate_oct_bounds(data, which_tree, octant);
    oct->octant_id = data->num_octants++;
    oct->mpirank = data->mpi->mpirank;
}


void p8est_setup(local_data_t * data){
    if(!data) return;
    
    p4est_init(NULL, SC_LP_DEFAULT);

    data->connectivity = p8est_connectivity_new_brick(p4est_box_size, p4est_box_size, p4est_box_size, 0, 0, 0);// p4est_box_size^3 non periodic
    if(!data->connectivity) {
        printf("ERROR: Failed to create connectivity\n");
        return;
    }

    data->p8est = p8est_new_ext(data->mpi->mpicomm, data->connectivity, 
                        0,//Min number of quadrants
                        0,//refine level
                        1,//uniform t/f
                        sizeof(octant_data_t),//small octant_local info
                        oct_init,//octant callback fxn
                        data);//pointer for global data struct
    
    if(!data->p8est) {
        printf("ERROR: Failed to create p8est\n");
        return;
    }

    p8est_partition(data->p8est, 0, NULL); // 0 and NULL for uniform weight distribution

    data->ghost = p8est_ghost_new(data->p8est, P8EST_CONNECT_FULL);
    if(!data->ghost) {
        printf("ERROR: Failed to create ghost\n");
        cleanup(data);
        return;
    }
    data->mesh = p8est_mesh_new(data->p8est, data->ghost, P8EST_CONNECT_FULL);
    if(!data->mesh) {
        printf("ERROR: Failed to create mesh\n");
        cleanup(data);
        return;
    }
    
    data->p8est->user_pointer = data;
}

void send_master_info(Master * S, local_data_t * data){
    if(!S || !data) return;
    //broadcast the master information to all other ranks
    //initalize sub_master object
    //populate sub_master
    //broadcast sub_master to all other ranks
}

void scale_particles(Master * S, local_data_t * data, double bound[6]){
    for(int i =0; i < S->NNodes(); i++){
        fdmnode * node = &S->GetNodeVec()[i];
        const double dx = bound[1] - bound[0];
        const double dy = bound[3] - bound[2];
        const double dz = bound[5] - bound[4];
        const double scaled_x = dx != 0.0 ? (node->x() - bound[0]) / dx * 100.0 : 0.0;
        const double scaled_y = dy != 0.0 ? (node->y() - bound[2]) / dy * 100.0 : 0.0;
        const double scaled_z = dz != 0.0 ? (node->z() - bound[4]) / dz * 100.0 : 0.0;
        node->set_scaled_pos(scaled_x, scaled_y, scaled_z);
    }
}
/*
void populate_inital_outgoing(Master * S, local_data * data){
    for(int i =0; i < S->NNodes(); i++){
        fdmnode * node = &S->GetNodeVec()[i];
        //search local octant


    }
    

}

*/

int main(int argc, char *argv[]) {

    Master S;
    int iOpt;
    /*
    if ( argc < 2 ) {
        cerr << "Usage: " << argv[0] << "[-n N] input.par\n"
             << "Options:\n"
             << "\t-n N: \t specify number of threads for parallel computing"<<endl;
        return 1;
    }
    while ((iOpt = getopt(argc, argv, "n:")) != -1) {
        switch (iOpt) {
            case 'n':
                S.iNthreads = atoi(optarg); // number of threads for parallel computing
                if ( S.iNthreads < 1 )
                    cerr << "Number of thread must be larger thant 1! (Input: " << S.iNthreads << ")" << endl;
                break;
            default:
                cerr << "Usage: " << argv[0] << "[-n N] input.par\n"
                << "Options:\n"
                << "\t-n N: \t specify number of threads for parallel computing"<<endl;
                return 1;
        }
    }
    */
    if(argc < 2){
        printf("No input file specified.\n");
        return 1;
    }
    // ^ skip processor count at the command line 
    //alternative to mpirun -n #processors
    
    //init p4est and MPI objects
    local_data_t data = {0};
    local_data_t * data_ptr = &data;

    mpi_context_t mpi_context;
    memset(&mpi_context, 0, sizeof(mpi_context));
    data_ptr->mpi = &mpi_context;
    data_ptr->num_octants = 0;

    mpi_set(argc, argv, data_ptr);
    p8est_setup(data_ptr);
    S.iNthreads = data_ptr->mpi->mpisize; //consitentcy
    

#ifdef USE_GAS
    cout << "USE_GAS open." << endl;
#endif
    
#ifdef USE_SELF
    cout << "USE_SELF open." << endl;
#endif
    /*
    omp_set_num_threads(S.iNthreads);
    cout << "GTasb3D running on " << S.iNthreads << " processors\n" << endl;
    */
#ifdef SEETIME
    clock_t t0,t1,t2;
    t0 = clock();
#endif

    if(data_ptr->mpi->mpirank == 0){
        /* Initializing the simulation */
        if ( S.InitSystem(argv[argc-1]) != 0 ) {
            cout << "Error occurred while initializing master." << endl;
            return 1;
        }
        S.Output();  /* output nodes' information */
        /*
        char debug_output[256];
        snprintf(debug_output, sizeof(debug_output), "debug_output_rank%d", data.mpi->mpirank);
        */
        //can do fprintf to debug_ouput
        double xm, ym, zm, xM, yM, zM;
        for(int i =0; i < S.NNodes(); i++){
            if(i == 0){
                xm = S.GetNodeVec()[i].x();
                ym = S.GetNodeVec()[i].y();
                zm = S.GetNodeVec()[i].z();
                xM = S.GetNodeVec()[i].x();
                yM = S.GetNodeVec()[i].y();
                zM = S.GetNodeVec()[i].z();
            }
            else{
                xm = min(xm, S.GetNodeVec()[i].x());
                ym = min(ym, S.GetNodeVec()[i].y());
                zm = min(zm, S.GetNodeVec()[i].z());
                xM = max(xM, S.GetNodeVec()[i].x());
                yM = max(yM, S.GetNodeVec()[i].y());
                zM = max(zM, S.GetNodeVec()[i].z());
            }
        }
        printf("The domain is x:(%.2f)-(%.2f) y:(%.2f)-(%.2f) z:(%.2f)-(%.2f)\n", xm, xM, ym, yM, zm, zM);
        
        //scale into [0-p4est_box_size]^3, much better for p4est partitioning
        double bounds[6] = {xm, xM, ym, yM, zm, zM};
        scale_particles(&S, data_ptr, bounds);
        
        for(int i =0; i < S.NNodes(); i++){
            printf("\nscaled)) X: %.3f, Y: %.3f, Z:% .3f", S.GetNodeVec()[i].get_scaled_X(), S.GetNodeVec()[i].get_scaled_Y(), S.GetNodeVec()[i].get_scaled_Z());
        }

        //populate_inital_outgoing(&S, data_ptr);
    }
    sc_MPI_Barrier(data_ptr->mpi->mpicomm);//hold 
    exit(0);



    //populate outgoing nodevecs
    //mpiIsend/recv


    //change particle data to include what rank it belongs to 
    //^includes neighbor list so that ranks know which to send and recv
    //
    

    send_master_info(&S, data_ptr);
    
    
#ifdef SEETIME
    t1 = clock();
    cout << "The input elapsed time: " << (double)(t1-t0) << " s." << endl;
#endif

    cout << "Simulation starts." << endl;

    while( S.CurrentStep() <= S.TotalSteps() ){
        if ( S.Run() )
            break; /* encounter running error */
        S.Output();  /* output nodes' information */
#ifdef SEETIME
        t2 = clock();
        cout<<"The elapsed time for "<<S.OutInterval()<<" runs: "<<(double)(t2-t1)<<" s."<<endl;
        t1 = clock();
#endif
    }
    cout << "Simulation ends." << std::endl;
#ifdef SEETIME
    t2 = clock();
    cout<<"The total elapsed time: "<<(double)(t2-t0)<<" s."<<endl;
#endif
    
    return 0;
}
