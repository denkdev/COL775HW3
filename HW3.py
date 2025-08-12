import time
import numpy as np
import statistics
import matplotlib.pyplot as plt

def naive_python_multiplication(A, B, n):
    result = [[0 for _ in range(n)] for _ in range(n)]
    
    # Perform matrix multiplication using three nested loops
    for i in range(n):           # For each row in A
        for j in range(n):       # For each column in B
            for k in range(n):   # For each element in the dot product
                result[i][j] += A[i][k] * B[k][j]
    
    return result


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

def numpy_multiplication(A, B, n):
    """NumPy matrix multiplication for comparison"""
    return np.dot(A, B)

    

n_values = [8, 16, 32, 64, 128]

# Naive implementation
naive_mean_gflops, naive_std_dev_gflops = analyze_matrix_multiplication(n_values, "Naive Python", naive_python_multiplication)

# Optimized NumPy implementation
np_mean_gflops, np_std_dev_gflops = analyze_matrix_multiplication(n_values, "Optimized NumPy", numpy_multiplication)

# CuPy implementation
    
# Simple plotting - just add this at the end
# Plot 1: Naive Python Performance
plt.figure(figsize=(8, 6))
plt.errorbar(n_values, naive_mean_gflops, yerr=naive_std_dev_gflops, 
            marker='o', capsize=5, color='blue')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Mean GFLOPS')
plt.title('Python Loop Performance')
plt.grid(True)
plt.savefig('naive_performance.png')
plt.show()

# Plot 2: NumPy Performance
plt.figure(figsize=(8, 6))
plt.errorbar(n_values, np_mean_gflops, yerr=np_std_dev_gflops, 
            marker='s', capsize=5, color='orange')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Mean GFLOPS')
plt.title('NumPy Performance')
plt.grid(True)
plt.savefig('numpy_performance.png')
plt.show()
    

