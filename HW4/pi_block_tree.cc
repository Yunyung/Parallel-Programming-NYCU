#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

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
    long long int recv_numOfhit = 0;
    
    unsigned int seed = world_rank * time(0);
    for (int i = 0;i < local_numOftosses;i++) {
        float x = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        float y = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

        if (x*x + y*y <= 1.0)
            local_numOfhit++;
    }

    // TODO: binary tree redunction
    // Here we assuming that we use a power-of-two number of processes.
    for (int round = 1;round < world_size;round <<= 1) {
        if ((world_rank & round) != 0) {
            MPI_Send(&local_numOfhit, 1, MPI_LONG_LONG_INT, world_rank - round, 0, MPI_COMM_WORLD);
            break;
        } else if ((world_rank + round < world_size)){
            MPI_Recv(&recv_numOfhit, 1, MPI_LONG_LONG_INT, world_rank + round, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_numOfhit += recv_numOfhit;
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
