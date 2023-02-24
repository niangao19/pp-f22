#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//


void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
    int numNodes = num_nodes(g);
    double equal_prob = 0.0;
    equal_prob = 1.0 / numNodes;
    double* score_old = (double*)malloc(sizeof(double) * numNodes);
    double* score_new = (double*)malloc(sizeof(double) * numNodes);

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

    
    int* v_outsize = (int*)malloc(sizeof(int) * numNodes);
    int* v_insize = (int*)malloc(sizeof(int) * numNodes);
    int* v_iszero = (int*)malloc(sizeof(int) * numNodes);
    int zeronum = 0;
    #pragma omp parallel for
    for( int i = 0; i < numNodes; i++ ) {
        score_old[i] = equal_prob;
        v_outsize[i] = outgoing_size(g, i);
        v_insize[i] = incoming_size(g, i);
        if( v_outsize[i]  == 0 ) {
            #pragma omp critical
            {
                v_iszero[zeronum] = i;
                zeronum++;
            }
        }
    } // for
    
    double num = (1.0-damping) / numNodes;
    bool converged = false;
    while (!converged) {
        double no_out_score = 0.0;
        double global_diff = 0.0;
        //  score_new[vi] += sum over all nodes v in graph with no outgoing edges   { damping * score_old[v] / numNodes }
        //  pass score_old
        #pragma omp parallel for reduction(+:no_out_score)
        for( int i = 0; i < zeronum; i++ ) {
            int j = v_iszero[i];
            no_out_score += score_old[j];
        } // for

        no_out_score = damping * no_out_score /numNodes;
        
        #pragma omp parallel for reduction(+:global_diff)
        for( int i = 0; i < numNodes; i++ ) {
            score_new[i] = 0.0;
            const Vertex*  in_vi_start = incoming_begin(g, i);
            int vnum = v_insize[i];
            for( int j = 0; j < vnum; j++ ) {
                int in_num = in_vi_start[j];
                score_new[i] += score_old[in_num] / v_outsize[in_num];
            } // for
            score_new[i] = ( damping * score_new[i] ) + num + no_out_score;
            global_diff += fabs(  score_new[i] - score_old[i] );;
        } // for

        converged = (global_diff < convergence);
        double* temp = score_old;
        score_old = score_new;
        score_new = temp;
    } // while
    
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
      solution[i] = score_old[i];
    }
}
