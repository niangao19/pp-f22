#include <stdio.h>
#include <thread>
#include <time.h>
#include "CycleTimer.h"

typedef struct
{
    float x0, x1, dx;
    float y0, y1, dy;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

static inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{
    //double  start =  clock();
    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrotSerial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrotSerial() to this file and
    // modify it to pursue a better performance.
    WorkerArgs *data = (WorkerArgs *)args;
    int numThreads = data->numThreads;
    int threadId = data->threadId;
    float dx = data->dx;
    float x0 = data->x0;
    float dy = data->dy;
    float y0 = data->y0;
    int* output = data->output;
    float x = x0;
    float y = y0;
    /*int numRows = data->height/numThreads;
    int startRow = numRows * threadId;
    mandelbrotSerial(
        data->x0, data->y0, data->x1, data->y1,
        data->width, data->height,
        startRow, numRows, data->maxIterations, data->output);*/
    for(unsigned int j = threadId; j < data->height; j += numThreads  ) {
        for (unsigned  int i = 0; i < data->width; ++i) {
          float x = x0 + i * dx;
          float y = y0 + j * dy;
          int index = (j * data->width + i);
          output[index] = mandel(x, y, data->maxIterations);
        }
    }

    //printf("Hello world from thread %d\n", args->threadId);
    //double end = clock();
    //double time = (end - start) /CLOCKS_PER_SEC;
    //printf("The %d thread : cost %f time\n", args->threadId, time );
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].dx = dx;
        args[i].dy = dy;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
