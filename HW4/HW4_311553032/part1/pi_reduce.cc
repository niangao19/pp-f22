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
    long long int *recv;
    // TODO: use MPI_Gather
    double x, y, distance_squared ;
    unsigned int seed = time(NULL);
    //srand( seed );
    for ( long long int toss = world_rank; toss < tosses; toss+=world_size ) {
        x = 2.0 * rand_r(&seed) / (RAND_MAX)-1.0;
        y = 2.0* rand_r(&seed) / (RAND_MAX)-1.0;
        distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            toss_in_circle++;
    } // for
    // TODO: use MPI_Reduce
    long long int total_in_circle = 0;
    MPI_Reduce( &toss_in_circle, &total_in_circle,1, MPI_LONG_LONG, MPI_SUM, 0 , MPI_COMM_WORLD );
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * total_in_circle / (( double ) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
