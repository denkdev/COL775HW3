import time
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.sparse as sparse

def scipy_linalg_multiplication(A, B, n):
    """SciPy linalg matrix multiplication"""
    return linalg.blas.dgemm(1.0, A, B)

def scipy_sparse_multiplication(A, B, n):
    """SciPy sparse matrix multiplication"""
    A_sparse = sparse.csr_matrix(A)
    B_sparse = sparse.csr_matrix(B)
    return A_sparse @ B_sparse

def analyze_matrix_multiplication(n_values, implementation_name, mult_func): 
    print(f"--- Benchmarking {implementation_name} ---")
    Mean_gflops = []    
    Std_dev_gflops = []
    for n in n_values:
        print(f"\nMatrix Size: {n} x {n}")
        matrix_a = np.random.rand(n, n)
        matrix_b = np.random.rand(n, n)
        
        gflops_values = []

        for i in range(30):
            start_time = time.time_ns()
            mult_func(matrix_a, matrix_b, n)
            end_time = time.time_ns()
            
            # Calculate execution time and GFLOPS
            execution_time = end_time - start_time
            # Handle case where execution_time might be 0 (very fast operations)
            if execution_time == 0:
                execution_time = 1  # Set to 1 nanosecond minimum
            # Correct FLOPS calculation: 2n³ for matrix multiplication
            flops = 2 * (n ** 3)
            gflops = flops / (execution_time)
            gflops_values.append(gflops)

        # Calculate mean and standard deviation
        mean_gflops = statistics.mean(gflops_values)
        std_dev_gflops = statistics.stdev(gflops_values)

        print(f"  Mean GFLOPS: {mean_gflops:.10f}")
        print(f"  Standard Deviation: {std_dev_gflops:.10f}")

        Mean_gflops.append(mean_gflops)
        Std_dev_gflops.append(std_dev_gflops)

    return Mean_gflops, Std_dev_gflops

n_values = [8, 16, 32, 64, 128]

# SciPy linalg BLAS implementation
scipy_linalg_mean_gflops, scipy_linalg_std_dev_gflops = analyze_matrix_multiplication(n_values, "SciPy linalg BLAS", scipy_linalg_multiplication)

# SciPy sparse matrix implementation
scipy_sparse_mean_gflops, scipy_sparse_std_dev_gflops = analyze_matrix_multiplication(n_values, "SciPy Sparse", scipy_sparse_multiplication)

# Simple plotting - just add this at the end
# Plot 1: SciPy linalg BLAS Performance
plt.figure(figsize=(8, 6))
plt.errorbar(n_values, scipy_linalg_mean_gflops, yerr=scipy_linalg_std_dev_gflops, 
            marker='^', capsize=5, color='blue')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Mean GFLOPS')
plt.title('SciPy linalg BLAS Performance')
plt.grid(True)
plt.savefig('scipy_linalg_performance.png')
plt.show()

# Plot 2: SciPy Sparse Performance
plt.figure(figsize=(8, 6))
plt.errorbar(n_values, scipy_sparse_mean_gflops, yerr=scipy_sparse_std_dev_gflops, 
            marker='s', capsize=5, color='green')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Mean GFLOPS')
plt.title('SciPy Sparse Performance')
plt.grid(True)
plt.savefig('scipy_sparse_performance.png')
plt.show()

