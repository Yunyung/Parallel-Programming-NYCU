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
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Count hit
    long long int local_numOfhit = 0;
    long long int local_numOftosses = tosses / world_size;
    long long int gather[world_size];

    unsigned int seed = world_rank * time(0);
    for (int i = 0;i < local_numOftosses;i++) {
        float x = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        float y = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

        if (x*x + y*y <= 1.0)
            local_numOfhit++;
    }

    // TODO: use MPI_Gather
    MPI_Gather(&local_numOfhit, 1, MPI_LONG_LONG_INT, gather, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i = 1;i < world_size;i++) {
            local_numOfhit += gather[i];
        }
        pi_result = 4.0 * (local_numOfhit / (double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
