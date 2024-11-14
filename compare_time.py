import torch
import time

# Define matrix dimensions
matrix_size = 10000
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Generate random matrices
cpu_matrix = torch.randn(matrix_size, matrix_size, device='cpu')
gpu_matrix = torch.randn(matrix_size, matrix_size, device=device)

# # CPU multiplication
start_time = time.time()
result_cpu = torch.mm(cpu_matrix, cpu_matrix)
cpu_time = time.time() - start_time

# GPU multiplication
start_time = time.time()
result_gpu = torch.mm(gpu_matrix, gpu_matrix)
gpu_time = time.time() - start_time

# Check if the results match
print("CPU and GPU results match:", torch.allclose(result_cpu, result_gpu.to('cpu')))

# Print performance results
print(f"CPU Time: {cpu_time:.4f} seconds")
print(f"GPU Time: {gpu_time:.4f} seconds")
