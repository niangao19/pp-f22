#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>


#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

/*
 view -1
 // attempt to add all neighbors to the new frontier
 for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
 {
     int outgoing = g->outgoing_edges[neighbor];

     if( __sync_bool_compare_and_swap( &distances[outgoing], NOT_VISITED_MARKER, distance ) ) {

         while( !__sync_bool_compare_and_swap(&sum, index, index+1)  )
             index = sum;
         new_frontier->vertices[index] = outgoing;
     }
 }
 
 view-2
 int sum = 0;
 int distance = distances[frontier->vertices[0]] +1;
 #pragma omp parallel for
 for (int i = 0; i < frontier->count; i++) {
     int node = frontier->vertices[i];
     int index;
     int start_edge = g->outgoing_starts[node];
     int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->outgoing_starts[node + 1];

     // attempt to add all neighbors to the new frontier
     for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
     {
         int outgoing = g->outgoing_edges[neighbor];

         if( __sync_bool_compare_and_swap( &distances[outgoing], NOT_VISITED_MARKER, distance ) ) {

             #pragma omp critical
             {
                 index = sum++;
             }
             new_frontier->vertices[index] = outgoing;
         }
     }
 }
 new_frontier->count = sum;
 */

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int distance)
{
    int num_node =  num_nodes(g);
    #pragma omp parallel
    {
        int* local_frontier = (int*)malloc(sizeof(int) * num_node);
        int index = 0;
        #pragma omp for schedule(dynamic, 512)
        for (int i = 0; i < frontier->count; i++) {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                               ? g->num_edges
                               : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if( distances[outgoing] != NOT_VISITED_MARKER ) continue;
                if( __sync_bool_compare_and_swap( &distances[outgoing], NOT_VISITED_MARKER, distance ) )
                {
                    local_frontier[index] = outgoing;
                    index++;
                }
            }
        } // for
        
        int count = new_frontier->count;
        while( !__sync_bool_compare_and_swap( &new_frontier->count, count, count+index) )
            count = new_frontier->count;
        memcpy(new_frontier->vertices+count, local_frontier, index*sizeof(int) );
    }

}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int distance = 1;
    int not_visit_node = num_nodes(graph)-1;
    while (frontier->count != 0 && not_visit_node > 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, distance);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        distance++;
        not_visit_node -= frontier->count;
    }
}

/*
 view-1
 // attempt to add all neighbors to the new frontier
for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
 int incoming = g->incoming_edges[neighbor];
 int in_distance = distances[incoming];
 if (in_distance != NOT_VISITED_MARKER && in_distance != distance) {
     distances[i] = distance;
     while( !__sync_bool_compare_and_swap(&sum, index, index+1)  )
         index = sum;
     new_frontier->vertices[index] = i;
     break;
 } // if
     
} // for
 view-2
 int nodenum = num_nodes(g);
 //int distance = distances[frontier->vertices[0]] +1;
 int sum = 0;
 #pragma omp parallel for reduction(+:sum)
 for (int i = 0; i < nodenum; i++) {
     if(distances[i] != NOT_VISITED_MARKER) continue;
     //int index = sum;
     int start_edge = g->incoming_starts[i];
     int end_edge = (i == g->num_nodes - 1)
         ? g->num_edges
         : g->incoming_starts[i + 1];
         
         // attempt to add all neighbors to the new frontier
     for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
         int incoming = g->incoming_edges[neighbor];
         int in_distance = distances[incoming];
         if (in_distance == distance-1) {
             distances[i] = distance;
             sum++;
             //new_frontier->vertices[index] = i;
             break;
         } // if
             
     } // for

 } // for
 frontier_num = sum;
 
 view-3
 int sum = 0;
 #pragma omp parallel reduction(+:sum)
 {
     int* local_frontier = (int*)malloc(sizeof(int) * frontier->count);
     int index = 0;
     #pragma omp for
     for (int i = 0; i < frontier->count; i++) {
         int node = frontier->vertices[i];
         if(distances[node] != NOT_VISITED_MARKER) continue;
         int start_edge = g->incoming_starts[node];
         int end_edge = (node == g->num_nodes - 1)
         ? g->num_edges
         : g->incoming_starts[node + 1];
         // attempt to add all neighbors to the new frontier
         for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
             int incoming = g->incoming_edges[neighbor];
             if ( distances[incoming] == distance-1 ) {
                 distances[node] = distance;
                 sum++;
                 break;
             } // if
         } // for
         
         if ( distances[node] != distance ) {
             local_frontier[index] = node;
             index++;
         } // if
     } // for
     
     #pragma omp critical
     {
         memcpy(new_frontier->vertices+new_frontier->count, local_frontier, index*sizeof(int) );
         new_frontier->count += index;
     }
 }
 
 frontier_num = sum;
 
 */
void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int distance, int & frontier_num)
{
    //int nodenum = num_nodes(g);
    //int distance = distances[frontier->vertices[0]] +1;
    int nodenum = num_nodes(g);
    //int distance = distances[frontier->vertices[0]] +1;
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic , 512)
    for (int i = 0; i < nodenum; i++) {
        if(distances[i] != NOT_VISITED_MARKER) continue;
        //int index = sum;
        int start_edge = g->incoming_starts[i];
        int end_edge = (i == g->num_nodes - 1)
            ? g->num_edges
            : g->incoming_starts[i + 1];
            
            // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            int in_distance = distances[incoming];
            if (in_distance == distance-1 ) {
                distances[i] = distance;
                sum++;
                //new_frontier->vertices[index] = i;
                break;
            } // if
                
        } // for

    } // for
    frontier_num = sum;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    
    /*#pragma omp parallel for
    for (int i = 1; i < graph->num_nodes; i++) {
        frontier->vertices[i-1] = i;
    }*/
    int not_visit_node = num_nodes(graph)-1;
    //frontier->count = num_nodes(graph)-1;
    // setup frontier with the root node
    //frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int distance = 1;
    int frontier_num = 1;
    while (frontier_num != 0 && not_visit_node != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, distance, frontier_num);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        /*vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;*/
        distance++;
        not_visit_node -=frontier_num;
        //printf("%d\n",  not_visit_node );
    }
}

void hybrid_top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int distance , int & frontier_num)
{
    int num_node =  num_nodes(g);
    //int distance = distances[frontier->vertices[0]] +1;
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic , 512)
    for (int i = 0; i < num_node; i++) {
        if(distances[i] != distance-1) continue;
        int start_edge = g->outgoing_starts[i];
        int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[i + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            if( distances[outgoing] != NOT_VISITED_MARKER ) continue;
            if( __sync_bool_compare_and_swap( &distances[outgoing], NOT_VISITED_MARKER, distance ) )
                sum++;
        }

    } // for
    frontier_num = sum;

}

void hybrid_bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int distance, int & frontier_num)
{
    //int nodenum = num_nodes(g);
    //int distance = distances[frontier->vertices[0]] +1;
    int nodenum = num_nodes(g);
    //int distance = distances[frontier->vertices[0]] +1;
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic, 512)
    for (int i = 0; i < nodenum; i++) {
        if(distances[i] != NOT_VISITED_MARKER) continue;
        //int index = sum;
        int start_edge = g->incoming_starts[i];
        int end_edge = (i == g->num_nodes - 1)
            ? g->num_edges
            : g->incoming_starts[i + 1];
            
            // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            int in_distance = distances[incoming];
            if (in_distance == distance-1 ) {
                distances[i] = distance;
                sum++;
                //new_frontier->vertices[index] = i;
                break;
            } // if
                
        } // for

    } // for
    frontier_num = sum;
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int distance = 1;
    int  frontier_num = 1;
    while (frontier_num != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        if( frontier_num < 100000 )
            hybrid_top_down_step(graph, frontier, new_frontier, sol->distances, distance, frontier_num);
        else
            hybrid_bottom_up_step( graph, frontier, new_frontier, sol->distances, distance, frontier_num);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        distance++;
    }
}
