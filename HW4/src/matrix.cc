#include <mpi.h>
#include <cstdio>
#include <cstdlib>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if( world_rank == 0 ) {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        int an = (*n_ptr) * (*m_ptr);
        *a_mat_ptr = ( int * )calloc( an , sizeof( int) );
        int bn = (*m_ptr) * (*l_ptr);
        *b_mat_ptr = ( int * )calloc( bn , sizeof( int) );

        for( int i = 0; i < an; i++ ) {
            scanf("%d", *a_mat_ptr+i );
        }
        
        for( int i = 0; i < (*m_ptr); i++ ) {
            for( int j = 0; j < (*l_ptr); j++ )
               scanf("%d", *b_mat_ptr + j*(*m_ptr)+i );
        }
    
    } // if
    
    MPI_Bcast( n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD );
    int local_n = (*n_ptr)/world_size;
    int an = local_n * (*m_ptr);
    int bn = (*m_ptr) * (*l_ptr);
    if( world_rank != 0  ) {
        *b_mat_ptr = ( int * )calloc( bn , sizeof( int) );
        *a_mat_ptr = ( int * )calloc( an , sizeof( int) );
    }
    int** local_a_mat = a_mat_ptr;
    MPI_Scatter(   *a_mat_ptr, an,MPI_INT, *a_mat_ptr, an, MPI_INT, 0, MPI_COMM_WORLD);
    if( world_rank == 0  )
        a_mat_ptr = local_a_mat;
    MPI_Bcast( *b_mat_ptr, bn, MPI_INT, 0, MPI_COMM_WORLD );
}


void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)  {
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int local_n = n/world_size;
    int bn = 0;
    int* result_local = ( int * )calloc( local_n  * l , sizeof( int) );
    // result[i][j] = result[i][k] * result[k][j] -> k = 0 ~ local_l
    for( int i = 0; i < local_n; i++ ) {
        for ( int k = 0; k < l; k++)  {
            result_local[ i * l + k] = 0;
            bn = k*m;
            for (  int j = 0 ; j < m; j++ ) {
                result_local[i* l + k] += a_mat[i*m+j]* b_mat[bn];
                bn++;
            }
        }
    }
    
    int *result ;
    if( world_rank == 0 )
        result = ( int * )calloc( n  * l , sizeof( int) );
    
    MPI_Gather( result_local, local_n * l, MPI_INT, result, local_n * l, MPI_INT, 0, MPI_COMM_WORLD );
    if( world_rank == 0 ) {
        
        int num = n % world_size;
        for( int i = n - num ; i < n; i++ ) {
            for ( int k = 0; k < l; k++)  {
                result[ i*l + k] = 0;
                bn = k*m;
                for (  int j = 0 ; j < m; j++   ) {
                    result[ i*l + k] += a_mat[i*m+j]* b_mat[bn];
                    bn++;
                } // for
            } // for
        } // for
        
        for( int i = 0; i < n; i++ ) {
            for ( int k = 0; k < l; k++) {
                printf( "%d ", result[i*l+k]);
            }
            printf( "\n");
        }
    } // if
    

    
}

void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}
