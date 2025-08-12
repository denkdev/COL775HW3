import matplotlib.pyplot as plt
import numpy as np

# Hardcoded values from C-Loop analysis
matrix_sizes = np.array([8, 16, 32, 64, 128])
mean_gflops = np.array([0.5, 1.2, 3.8, 12.5, 45.2])  # Example values
std_dev_gflops = np.array([0.1, 0.3, 0.8, 2.1, 5.3])  # Example values

# Create the plot
plt.figure(figsize=(8, 6))
plt.errorbar(matrix_sizes, mean_gflops, yerr=std_dev_gflops, 
            marker='D', capsize=5, color='purple')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Mean GFLOPS')
plt.title('C-Loop Performance')
plt.grid(True)
plt.savefig('c_loop_performance.png')
plt.show() 