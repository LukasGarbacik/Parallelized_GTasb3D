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
#include "bodyphysics.hpp"
#include "split.hpp"
#include <ctime>
#include <time.h>
#include <algorithm>
#include <cstring>


void cleanup(local_data_t * data){
    if(!data) return;

    if(data->mesh) {
        p8est_mesh_destroy(data->mesh);
        data->mesh = NULL;
    }
    if(data->ghost) {
        p8est_ghost_destroy(data->ghost);
        data->ghost = NULL;
    }
    if(data->p8est) {
        p8est_destroy(data->p8est);
        data->p8est = NULL;
    }
    if(data->connectivity) {
        p8est_connectivity_destroy(data->connectivity);
        data->connectivity = NULL;
    }
    
    if(data->mpi) {
        sc_MPI_Barrier(data->mpi->mpicomm);
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

//possibly add epison comparison in case the floats are the issue
void populate_oct_bounds(local_data_t * data, p4est_topidx_t which_tree, p8est_quadrant_t *octant){
    if(!data || !octant || !octant->p.user_data) return;
    
    octant_data_t *oct = (octant_data_t *) octant->p.user_data;

    int x = octant->x;
    int y = octant->y;
    int z = octant->z;

    double vertex[3];
    p8est_qcoord_to_vertex(data->connectivity, which_tree, x, y, z, vertex);

    oct->bounds.x_min = vertex[0];
    oct->bounds.x_max = vertex[0] + 1.0;
    oct->bounds.y_min = vertex[1];
    oct->bounds.y_max = vertex[1] + 1.0;
    oct->bounds.z_min = vertex[2];
    oct->bounds.z_max = vertex[2] + 1.0;
}

bool in_bounds(vector3d scaled_pos, octant_bounds_t bounds){
    return (scaled_pos.x() >= bounds.x_min && scaled_pos.x() <= bounds.x_max &&
            scaled_pos.y() >= bounds.y_min && scaled_pos.y() <= bounds.y_max &&
            scaled_pos.z() >= bounds.z_min && scaled_pos.z() <= bounds.z_max);
}


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

void scale_particles(Master * S, local_data_t * data, double bound[6]){
    //global bounds for min/max of entire domain
    for(int i =0; i < S->NNodes(); i++){
        fdmnode * node = &S->GetNodeVec()[i];
        const double dx = bound[1] - bound[0];
        const double dy = bound[3] - bound[2];
        const double dz = bound[5] - bound[4];
        const double scaled_x = dx != 0.0 ? (node->x() - bound[0]) / dx * p4est_box_size : 0.0;
        const double scaled_y = dy != 0.0 ? (node->y() - bound[2]) / dy * p4est_box_size : 0.0;
        const double scaled_z = dz != 0.0 ? (node->z() - bound[4]) / dz * p4est_box_size : 0.0;
        node->set_scaled_pos(scaled_x, scaled_y, scaled_z);
    }
}

void print_bounds(int rank, octant_bounds_t *bounds){
    if(!bounds) {
        printf("ERROR: print_bounds called with NULL pointer\n");
        return;
    }
    printf("Octant Bounds for rank %d:\n", rank);
    printf("  X: [%.6f, %.6f]\n", bounds->x_min, bounds->x_max);
    printf("  Y: [%.6f, %.6f]\n", bounds->y_min, bounds->y_max);
    printf("  Z: [%.6f, %.6f]\n", bounds->z_min, bounds->z_max);
}
void print_particle(int rank, int bound_num, fdmnode * particle){
    printf("Rank %d, oct %d\n", rank, bound_num);
    printf("particle %d: %.3f %.3f %.3f\n\n", particle->nID(), particle->get_scaled_X(), particle->get_scaled_Y(), particle->get_scaled_Z());
}


void populate_maps(Master * S, local_data_t * data){
    if (!data || !data->mpi) return;

    std::vector<octant_bounds_t> local_bounds;
    for (p4est_locidx_t i = 0; i < data->mesh->local_num_quadrants; ++i) {
        p8est_quadrant_t *oct = p8est_mesh_quadrant_cumulative(data->p8est, data->mesh, i, NULL, NULL);
        if(!oct) exit(1);
        
        octant_data_t * data = (octant_data_t *) oct->p.user_data;
        if(!data) exit(1);
        local_bounds.push_back(data->bounds);
    }
    printf("getting HERE1");
    for(int i =0; i < S->NNodes(); i++){
        fdmnode * node = &S->GetNodeVec()[i];
        if(node->type() < 2) data->needed.push_back(node->nID() + S->GetNplayer());
        if(node->type() == 1 && node->nID() >= S->GetNplayer()) data->needed.push_back(node->nID() - S->GetNplayer());
        else{
            for(int j = 0; j < node->get_num_neighbors(); j++){
                data->needed.push_back(node->get_neighbor_id(j));
            }
        }
        for(int j = 0; j < local_bounds.size(); j++){
            if(in_bounds(node->get_scaled_vPos(), local_bounds[j])){
                data->local_nodes[node->nID()] = node;
                break;
            }
        }
    }
    std::sort(data->needed.begin(), data->needed.end());
    auto it = std::unique(data->needed.begin(), data->needed.end());
    
    data->needed.erase(it, data->needed.end());
    //remove local nodes from globally needed list
    for(int i = 0; i < data->needed.size(); i++){
        if(data->local_nodes.count(data->needed[i]) == 1){
            data->needed.erase(data->needed.begin() + i);
            i--;
        }
    }
    if(data->mpi->mpirank == 0){
        printf("rank %d needed size: %d, local_map_size: %d\n", data->mpi->mpirank, data->needed.size(), data->local_nodes.size());
    }
    else{
        printf("RANK %d needed size: %d, local_map_size: %d\n", data->mpi->mpirank, data->needed.size(), data->local_nodes.size());
    }
}

void setup_outgoing(Master * S, local_data_t * data){
    if (!data || !data->mpi) return;

    //allgather counts of needed from each rank
    std::vector<int> needed_counts(data->mpi->mpisize, 0);
    int need_size = data->needed.size();
    sc_MPI_Allgather(
        &need_size, 1, sc_MPI_INT,
        needed_counts.data(), 1, sc_MPI_INT, data->mpi->mpicomm
        );
        

    std::vector<std::vector<unsigned int>> needed_matrix(data->mpi->mpisize);
    needed_matrix[data->mpi->mpirank] = data->needed;

    for (int rank = 0; rank < data->mpi->mpisize; rank++) {
        int count = needed_counts[rank];
        if (rank == data->mpi->mpirank) {//send vector to all ranks
            if (count > 0) {
                for (int out_rank = 0; out_rank < data->mpi->mpisize; out_rank++) {
                    if (out_rank == data->mpi->mpirank) continue;
                    sc_MPI_Send(
                        data->needed.data(),
                        count,
                        sc_MPI_UNSIGNED,
                        out_rank,
                        100 + rank,
                        data->mpi->mpicomm
                    );
                }
            }
        } else {//recv vector from all other ranks
            if (count > 0) {
                needed_matrix[rank].resize(count);
                sc_MPI_Recv(
                    needed_matrix[rank].data(),
                    count,
                    sc_MPI_UNSIGNED,
                    rank,
                    100 + rank,
                    data->mpi->mpicomm,
                    sc_MPI_STATUS_IGNORE
                );
            }
        }
    }


    // 3) determine what ids need to be sent to each rank and how many
    //    For each rank r, look at the IDs it needs and see which ones we own locally.
    data->outgoing.resize(data->mpi->mpisize);
    for (int rank = 0; rank < data->mpi->mpisize; rank++) {
        if (rank == data->mpi->mpirank) continue; // no need to send to self
        const auto &incoming = needed_matrix[rank];
        for (unsigned int nid : incoming) {
            if (data->local_nodes.count(nid) == 1) {
                fdmnode * node = data->local_nodes[nid];
                data->outgoing[rank].push_back(
                    temp_mpi_data_t{
                        nid,
                        &node->Temp(),
                        &node->Conductivity()
                    }
                );
            }
        }
    }
    if(data->mpi->mpirank == 1){
       for(int rank = 0; rank < data->mpi->mpisize; rank++){
            printf("rank %d: # outgoing:%d\n", data->mpi->mpirank, data->outgoing[rank].size());   
        }
        printf("rank %d: example: nid:%d, temp:%.3f, cond:%.3f\n", data->mpi->mpirank, data->outgoing[0][0].nid, *data->outgoing[0][0].temp, *data->outgoing[0][0].cond);
    }
}

void comm_neighbors_init(Master * S, local_data_t * data){
    if (!data || !data->mpi) return;

    //outgoing counts per rank
    std::vector<int> send_counts(data->mpi->mpisize, 0);
    for (int dst_rank = 0; dst_rank < data->mpi->mpisize; dst_rank++){
        if (dst_rank == data->mpi->mpirank) continue;
        send_counts[dst_rank] = static_cast<int>(data->outgoing[dst_rank].size());
    }


    /* debug info
    int total_send = 0;
    for (int dst_rank = 0; dst_rank < data->mpi->mpisize; ++dst_rank) {
        total_send += send_counts[dst_rank];
    }
    printf("comm_neighbors_init rank %d: total outgoing entries = %d\n", data->mpi->mpirank, total_send);
    for (int dst_rank = 0; dst_rank < data->mpi->mpisize; ++dst_rank) {
        if (send_counts[dst_rank] > 0) {
            printf("  -> to rank %d: %d entries\n", dst_rank, send_counts[dst_rank]);
        }
    }
        */

    //alltoall works populating the incoming counts
    data->incoming_counts.assign(data->mpi->mpisize, 0);
    sc_MPI_Alltoall(
        send_counts.data(), 1, sc_MPI_INT,
        data->incoming_counts.data(), 1, sc_MPI_INT,
        data->mpi->mpicomm
    );

    //offset vector
    data->offsets.assign(data->mpi->mpisize, 0);
    int total_incoming = 0;
    for (int src_rank = 0; src_rank < data->mpi->mpisize; src_rank++) {
        data->offsets[src_rank] = total_incoming;
        total_incoming += data->incoming_counts[src_rank];
    }

    // Debug: summarize incoming sizes per rank
    printf("comm_neighbors_init rank %d: total incoming entries = %d\n", data->mpi->mpirank, total_incoming);
    for (int src = 0; src < data->mpi->mpisize; ++src) {
        if (data->incoming_counts[src] > 0) {
            printf("  <- from rank %d: %d entries\n", src, data->incoming_counts[src]);
        }
    }

    //Sending node data via mpi (init)

    //flattened buffer for all global neighbors
    data->recv_buffer.resize(total_incoming);

    std::vector<std::vector<byte_node_t>> send_buffers(data->mpi->mpisize);
    //reqs populated by both send and recv.
    //at max 2 * (data->mpi->mpisize - 1) comms
    std::vector<sc_MPI_Request> reqs;
    reqs.reserve(2 * (data->mpi->mpisize - 1));

    for (int peer = 0; peer < data->mpi->mpisize; peer++){
        if (peer == data->mpi->mpirank) continue;

        int scount = send_counts[peer];
        int rcount = data->incoming_counts[peer];

        //Recv buffer from rank peer
        if (data->incoming_counts[peer] > 0) {
            byte_node_t * recv_ptr = data->recv_buffer.data() + data->offsets[peer];
            sc_MPI_Request rreq;
            sc_MPI_Irecv(
                recv_ptr,
                data->incoming_counts[peer] * (int) sizeof(byte_node_t),
                sc_MPI_BYTE,
                peer,
                400,
                data->mpi->mpicomm,
                &rreq
            );
            reqs.push_back(rreq);
        }

        // Pack and send our data to this peer
        if (send_counts[peer] > 0) {
            send_buffers[peer].resize(send_counts[peer]);
            for (int i = 0; i < send_counts[peer]; i++) {
                const temp_mpi_data_t &src = data->outgoing[peer][i];
                send_buffers[peer][i].nid  = data->outgoing[peer][i].nid;
                send_buffers[peer][i].temp = *data->outgoing[peer][i].temp;
                send_buffers[peer][i].cond = *data->outgoing[peer][i].cond;
            }
            sc_MPI_Request sreq;
            sc_MPI_Isend(
                send_buffers[peer].data(),
                send_counts[peer] * (int) sizeof(byte_node_t),
                sc_MPI_BYTE,
                peer,
                400,
                data->mpi->mpicomm,
                &sreq
            );
            reqs.push_back(sreq);
        }
    }

    if (!reqs.empty()) {
        sc_MPI_Waitall((int) reqs.size(), reqs.data(), sc_MPI_STATUSES_IGNORE);
    }
    data->global_data.clear();

    for (auto &node : data->recv_buffer) {
        temp_mpi_data_t &inserted_node = data->global_data[node.nid];
        inserted_node.nid  = node.nid;
        inserted_node.temp = &node.temp;
        inserted_node.cond = &node.cond;
    }
    /*
    // Debug: print an example incoming node on one rank (mirrors outgoing debug)
    if (data->mpi->mpirank == 1 && total_incoming > 0) {
        const temp_mpi_data_t &ex = data->incoming[5];
        printf("comm_neighbors_init rank %d example incoming: nid:%u, temp:%.3f, cond:%.3f\n",
               data->mpi->mpirank, ex.nid, *ex.temp, *ex.cond);
    }

    // Debug: final size of flattened incoming buffer on this rank
    printf("comm_neighbors_init rank %d: data->incoming.size() = %zu\n",
           data->mpi->mpirank, data->incoming.size());
    */
}

//iterative function

void update_neighbors(Master * S, local_data_t * data){
    if (!data || !data->mpi) return;

    //temp buffers
    std::vector<std::vector<byte_node_t>> send_buffers(data->mpi->mpisize);
    std::vector<sc_MPI_Request> reqs;
    reqs.reserve(2 * (data->mpi->mpisize - 1));

    // Post all receives and sends
    for (int peer = 0; peer < data->mpi->mpisize; peer++) {
        if (peer == data->mpi->mpirank) continue;

        //recv directly into flattened buffer previously used
        if (data->incoming_counts[peer] > 0) {
            byte_node_t *recv_ptr = data->recv_buffer.data() + data->offsets[peer];
            sc_MPI_Request rreq;
            sc_MPI_Irecv(
                recv_ptr,
                static_cast<int>(data->incoming_counts[peer]) * (int) sizeof(byte_node_t),
                sc_MPI_BYTE,
                peer,
                400,
                data->mpi->mpicomm,
                &rreq
            );
            reqs.push_back(rreq);
        }

        //pull from outgoing pointers and do byte sends
        if (data->outgoing[peer].size() > 0) {
            send_buffers[peer].resize(data->outgoing[peer].size());
            for (int i = 0; i < data->outgoing[peer].size(); i++) {
                send_buffers[peer][i].nid  = data->outgoing[peer][i].nid;
                send_buffers[peer][i].temp = *(data->outgoing[peer][i].temp);
                send_buffers[peer][i].cond = *(data->outgoing[peer][i].cond);
            }

            sc_MPI_Request sreq;
            sc_MPI_Isend(
                send_buffers[peer].data(),
                static_cast<int>(data->outgoing[peer].size()) * (int) sizeof(byte_node_t),
                sc_MPI_BYTE,
                peer,
                400,
                data->mpi->mpicomm,
                &sreq
            );
            reqs.push_back(sreq);
        }
    }

    //wait for all comm to finish
    if (!reqs.empty()) {
        sc_MPI_Waitall((int) reqs.size(), reqs.data(), sc_MPI_STATUSES_IGNORE);
    }
}

int main(int argc, char *argv[]) {

    Master S;
    int iOpt;

    if(argc < 2){
        printf("No input file specified.\n");
        return 1;
    }
    // ^ skip processor count at the command line 
    //alternative to mpirun -n #processors
    
    //init p4est and MPI objects
    local_data_t data = {0};
    local_data_t * data_ptr = &data;
    //expose localdata
    g_data = data_ptr;

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
    cout << "GTasb3D running on " << data_ptr->mpi->mpisize << " processors\n" << endl;
#ifdef SEETIME
    clock_t t0,t1,t2;
    t0 = clock();
#endif

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
    
    //for(int i =0; i < S.NNodes(); i++){
    //    printf("\nscaled)) X: %.3f, Y: %.3f, Z:% .3f", S.GetNodeVec()[i].get_scaled_X(), S.GetNodeVec()[i].get_scaled_Y(), S.GetNodeVec()[i].get_scaled_Z());
    //}

    populate_maps(&S, data_ptr);

    setup_outgoing(&S, data_ptr);

    comm_neighbors_init(&S, data_ptr);

    sc_MPI_Barrier(data_ptr->mpi->mpicomm);

#ifdef SEETIME
    t1 = clock();
    cout << "The input elapsed time: " << (double)(t1-t0) << " s." << endl;
#endif

    cout << "Simulation starts." << endl;

    while( S.CurrentStep() <= S.TotalSteps() ){
        if ( S.Run() )
            break; /* encounter running error */
        //iterative MPI send of neighbor data
        update_neighbors(&S, data_ptr);

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
