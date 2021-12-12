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

    MPI_Win win;

    // TODO: MPI init
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Count hit
    long long int local_numOfhit = 0;
    long long int local_numOftosses = tosses / world_size;
    long long int global_numOfhit;

    unsigned int seed = world_rank * time(0);
    for (int i = 0;i < local_numOftosses;i++) {
        float x = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        float y = ((float) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

        if (x*x + y*y <= 1.0)
            local_numOfhit++;
    }

    long long int *gather;
    if (world_rank == 0)
    {
        // Master
        MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &gather);
        *gather = 0;
        MPI_Win_create(gather, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&local_numOfhit, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT,
                    MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        *gather += local_numOfhit; 
        pi_result = 4.0 * (*gather / (double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
        MPI_Free_mem(gather);
    }

    MPI_Finalize();
    return 0;
}