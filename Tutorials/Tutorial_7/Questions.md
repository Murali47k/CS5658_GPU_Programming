## Tutorial 7 Questions

### Q1 : Implement 1024x1024 Matrix Multiplication using GPU cores without Tensor cores.

Implements 1024×1024 matrix multiplication using integer elements and ordinary CUDA GPU cores, without Tensor Cores or shared-memory tiling. Profile the execution on the Bhavani cluster using NVIDIA Nsight Systems (`nsys`) and submit the profiling results.

**Files:** `matmulpr.cu`, `submit.sh`

```bash
# run to create executable
nvcc matmulpr.cu -o matmulpr

# run to create output files
sbatch submit.sh
```

**Output Files:** `matmul_profile.nsys-rep` , `matmul_profile.sqlite` ,`job.13286.out` 

---