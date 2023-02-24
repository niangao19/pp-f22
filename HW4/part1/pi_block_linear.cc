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

    // TODO: Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    // TODO: Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    long long int toss_in_circle = 0;
    //long long int nptoss = tosses / world_size;
    double x, y, distance_squared ;
    unsigned int seed = time(NULL)*world_rank;
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
        // TODO: handle workers

        MPI_Send(&toss_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } // if
    else if (world_rank == 0)
    {
        // TODO: master
        
        long long int np_in_circle = 0;
        for( int i = 1; world_size > i; i++ ) {
            MPI_Recv(&np_in_circle, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            toss_in_circle += np_in_circle;
        } // for
    } // else if

    if (world_rank == 0)
    {
        // TODO: process PI result
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
