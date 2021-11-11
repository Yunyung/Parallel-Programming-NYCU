#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_mutex_t mutexSum;

typedef struct
{
    int start;
    int end;
    long long int *numOfhit; // Used to access the global value of the number of hit 
} Arg_t; // The arguments pass to pthread_create() function

void* calpi(void *args) {

    Arg_t *arg = (Arg_t *)args;
    int start = arg->start;
    int end = arg->end;

    long long int *numOfhit = arg->numOfhit;
    long long int local_numOfhit = 0;
    
    unsigned int seed = 777;
    
    for (int i = start;i < end;i++) {
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        
        double dist = x*x + y*y;
        if (dist <= 1.0)
            local_numOfhit++;
    }

    pthread_mutex_lock(&mutexSum);
    *numOfhit += local_numOfhit;
    pthread_mutex_unlock(&mutexSum);

    pthread_exit((void *)0);
}

int main(int argc, char** argv) {
    // parse arguments
    int numOfthread = atoi(argv[1]);
    long long int numOftosses = atoll(argv[2]);

    int perPartitionSize = (numOftosses / numOfthread);

    pthread_t threads[numOfthread];
    Arg_t arg[numOfthread];

    // Initialize mutex lock
    pthread_mutex_init(&mutexSum, NULL);

    // Set pthread attribute and let pthread joinable
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    long long int *numOfhit = (long long int *) malloc(sizeof(*numOfhit));
    *numOfhit = 0; // Init the number of hit as zero

    for (int i = 0;i < numOfthread;i++) {
        /* Set argueemnts to each thread */
        arg[i].start = perPartitionSize * i;
        arg[i].end = perPartitionSize * (i+1);
        arg[i].numOfhit = numOfhit; // All threads point to the same memory address (Shared by all threads)  

         // Create a new thread and run calPi() function with correspoding arg[i] arguments
        pthread_create(&threads[i], &attr, calpi, (void *) &arg[i]);
    }
    
    /* Free attribute*/
    pthread_attr_destroy(&attr);
    
    /* Wait for the other threads*/
    void *status;
    for (int i = 0;i < numOfthread;i++) {
        pthread_join(threads[i], &status);
    }

    pthread_mutex_destroy(&mutexSum);

    double pi = 4 * ((*numOfhit) / (double)numOftosses);
    printf("%.7lf\n", pi);

    return 0;
}