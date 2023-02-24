///
//  main.cpp
//  pi
//
//  Created by 年晧鳴 on 2022/9/15.
//
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

pthread_mutex_t mutexpi;
typedef struct
{
   long long int tossnum;
   long long int  *number_in_circle;
} Arg; // 傳入 thread 的參數型別




void* thread_pi(void *arg){
    Arg *data = (Arg *)arg;
    long long int number_of_tosses = data->tossnum;
    long long int *number_in_circle = data->number_in_circle;
    long long int local_in_circle = 0;
    float x[8];
    float y[8];
    float dist[8];
    __m256 randmax = _mm256_set1_ps(2147483647);
    __m256 spac = _mm256_set1_ps(2.0);
    __m256 min = _mm256_set1_ps(-1.0);
    unsigned int seed = time(NULL);
    __m256 xvec, yvec, powx_vec, powy_vec, dist_vec;
    for ( long long int toss = 0; toss < number_of_tosses; toss= toss+8) {
        /*double x = 2.0* rand() / (RAND_MAX)-1.0;
        double y = 2.0* rand() / (RAND_MAX)-1.0;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            local_in_circle++;*/
        for( int i = 0; i< 8; i++ ) {
            x[i] = rand_r(&seed);
            y[i] = rand_r(&seed);
        } //for
        //x = ( max - min ) * rand() / (RAND_MAX+1.0)+min
        //y = ( max - min ) * rand() / (RAND_MAX+1.0)+min
        xvec = _mm256_loadu_ps(&x[0]);
        yvec = _mm256_loadu_ps(&y[0]);
        xvec = _mm256_mul_ps( xvec, spac);
        yvec = _mm256_mul_ps( yvec, spac);
        xvec = _mm256_div_ps( xvec, randmax);
        yvec = _mm256_div_ps( yvec, randmax);
        xvec = _mm256_add_ps( xvec, min);
        yvec = _mm256_add_ps( yvec, min);

        // distance_squared = x * x + y * y;
        powx_vec = _mm256_mul_ps( xvec, xvec);
        powy_vec = _mm256_mul_ps(yvec, yvec);
        dist_vec = _mm256_add_ps( powx_vec, powy_vec );
        //if ( distance_squared <= 1)
        //    local_in_circle++;
        _mm256_storeu_ps(&dist[0], dist_vec);
        for( int i = 0; i< 8; i++ ) {
            if ( dist[i] <= 1)
                local_in_circle++;
        } //for
    } // for
    pthread_mutex_lock(&mutexpi);
    *(number_in_circle) += local_in_circle;
    pthread_mutex_unlock(&mutexpi);
}




int main(int argc, const char * argv[]) {
    // 初始化互斥鎖
    pthread_mutex_init(&mutexpi, NULL);
    // 設定 pthread 性質是要能 join
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int NUM_THREADS = atoi(argv[1]);
    long long int number_of_tosses = atoi(argv[2]);
    // 每個 thread 都可以存取的 PI
    // 因為不同 thread 都要能存取，故用指標
    long long int *number_in_circle = (long long int *)malloc(sizeof(*number_in_circle));
    *number_in_circle = 0;
    pthread_t threads[NUM_THREADS];
    Arg arg[NUM_THREADS];
    long long int threadtoss = number_of_tosses/NUM_THREADS;
        
    for ( int i = 0; i < NUM_THREADS; i++) {
        arg[i].tossnum = threadtoss;
        arg[i].number_in_circle = number_in_circle;
        pthread_create( &threads[i], &attr, thread_pi, (void *)&arg[i]  );
    } // for
    
    pthread_attr_destroy(&attr);
    if( number_of_tosses%NUM_THREADS != 0 )
        number_of_tosses -= number_of_tosses%NUM_THREADS;
    if(  threadtoss%8 != 0 )
        number_of_tosses += ( 8 - (threadtoss%8) ) * NUM_THREADS;
    for ( int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    } // for
    pthread_mutex_destroy(&mutexpi);
    *(number_in_circle) = *(number_in_circle) *4;
    double pi =  *(number_in_circle) /(( double ) number_of_tosses);
    printf( "%f\n", pi );
    return 0;
}

