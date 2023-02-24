#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    // TODO: Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    // TODO: Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    long long int toss_in_circle = 0;
//    long long int nptoss = tosses / world_size;
    double x, y, distance_squared ;
    long long int *recv_data = new long long int[world_size-1];
    unsigned int seed = time(NULL);
    //srand( seed );
    for ( long long int toss = world_rank; toss < tosses; toss+=world_size ) {
        x = 2.0 * rand_r(&seed) / (RAND_MAX)-1.0;
        y = 2.0* rand_r(&seed) / (RAND_MAX)-1.0;
        distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            toss_in_circle++;
    } // for
    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&toss_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        for( int i = 1; world_size > i; i++ ) {
            MPI_Irecv( &recv_data[i-1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i-1] );
        } // for
        MPI_Waitall( world_size-1, requests, MPI_STATUS_IGNORE);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        for( int i = 1; world_size > i; i++ ) {
            toss_in_circle += recv_data[i-1];
        } // for
        pi_result = 4 * toss_in_circle / (( double ) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
