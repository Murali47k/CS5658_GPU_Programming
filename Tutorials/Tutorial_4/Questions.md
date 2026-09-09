## Tutorial 2 Questions

---

#### Q1 : Basic Matrix Multiplication with Tiling

Write a CUDA C/C++ program that implements matrix multiplication with tiling for a **4x4 matrix** using a **tile size of 2**. Use hardcoded values for the input matrices. The program should verify the GPU result against a CPU reference implementation.

**File:** `matrix_mul.cu`

##### **Expected Output**

```text
Input matrices (4x4):
Matrix A:
    1.00     2.00     3.00     4.00 
    5.00     6.00     7.00     8.00 
    9.00    10.00    11.00    12.00 
   13.00    14.00    15.00    16.00 

Matrix B:
   16.00    15.00    14.00    13.00 
   12.00    11.00    10.00     9.00 
    8.00     7.00     6.00     5.00 
    4.00     3.00     2.00     1.00 


GPU Result (using tiling):
Matrix C (GPU):
   80.00    70.00    60.00    50.00 
  240.00   214.00   188.00   162.00 
  400.00   358.00   316.00   274.00 
  560.00   502.00   444.00   386.00 

CPU Reference result:
Matrix C (CPU):
   80.00    70.00    60.00    50.00 
  240.00   214.00   188.00   162.00 
  400.00   358.00   316.00   274.00 
  560.00   502.00   444.00   386.00 

Results match!
```

---

#### Q2 : Large Matrix Multiplication with Performance Comparison

Write a CUDA C/C++ program that implements matrix multiplication with tiling for a **512x512 matrix** using a **tile size of 16**. Use random values (0-9) to initialize the matrices. Compare and report the execution time between GPU and CPU implementations.

**File:** `matrix_mul_large.cu`


##### **Expected Output**

```text
Matrix Size: 512 x 512
Tile Size: 16 x 16
CPU Time: 1.3902 seconds
GPU Time: 0.0266 seconds
Speedup: 52.31x
Results match!
```
---
