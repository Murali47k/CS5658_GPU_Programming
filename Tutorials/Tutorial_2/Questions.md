## Tutorial 2 Questions

---

#### Q1 : SSSP on GPU

Implement **Single-Source Shortest Path (SSSP)** on the GPU with the graph stored in **CSR (Compressed Sparse Row) format**.

The graph can be read from the hard disk into **CPU RAM** and stored in CSR format.

First, implement a **serial SSSP algorithm on the CPU** and verify its correctness. Then, implement the **SSSP algorithm on the GPU** and compare the results with the CPU implementation.

**File:** `bf_serial.cpp` & `bf_cuda.cu`

---

#### Expected Output

For the given graph, the expected shortest-path distances from source vertex **0** are:

```text
Vertex 0: 0
Vertex 1: 4
Vertex 2: 2
Vertex 3: 3
Vertex 4: 5
Vertex 5: 8
Vertex 6: 8
Vertex 7: 10
Vertex 8: 13
Vertex 9: 14
```