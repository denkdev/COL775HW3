#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply_c(double* A, double* B, double* C, int n) {
    // Initialize result matrix C to zero
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }
    
    // Perform matrix multiplication using three nested loops
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Function to generate random matrix
void generate_random_matrix(double* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

// Function to print matrix (for debugging)
void print_matrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int n = 64;  // Matrix size
    
    // Allocate memory for matrices
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    
    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Generate random matrices A and B
    generate_random_matrix(A, n);
    generate_random_matrix(B, n);
    
    printf("Matrix A (%dx%d):\n", n, n);
    print_matrix(A, n);
    
    printf("Matrix B (%dx%d):\n", n, n);
    print_matrix(B, n);
    
    // Perform matrix multiplication
    clock_t start = clock();
    matrix_multiply_c(A, B, C, n);
    clock_t end = clock();
    
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Result Matrix C (%dx%d):\n", n, n);
    print_matrix(C, n);
    
    printf("Matrix multiplication completed in %.6f seconds\n", cpu_time_used);
    
    // Calculate FLOPS
    double flops = 2.0 * n * n * n;  // 2n³ for matrix multiplication
    double gflops = flops / (cpu_time_used * 1e9);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Free allocated memory
    free(A);
    free(B);
    free(C);
    
    return 0;
} 