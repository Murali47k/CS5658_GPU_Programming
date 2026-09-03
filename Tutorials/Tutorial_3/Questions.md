## Tutorial 3 Questions

---

#### Q1 : BFS on GPU

Implement Breadth-First Search (BFS) on the GPU with the graph stored in CSR format.

First, implement the BFS algorithm on the GPU using CUDA. Each CUDA thread processes one vertex. The BFS is performed level by level, where only vertices discovered at the current level process their neighbours.

The implementation should not use atomic instructions or locks. Instead, unvisited vertices are marked by checking whether their distance is `-1`.

The source vertex is **vertex 0**.

**File:** `bfs_cuda.cu`

---

#### Expected Output

For the given 15-node graph, the expected shortest-path distances from source vertex **0** are:

```text
Shortest distances from vertex 0:

Vertex 0: 0
Vertex 1: 1
Vertex 2: 1
Vertex 3: 2
Vertex 4: 2
Vertex 5: 3
Vertex 6: 3
Vertex 7: 3
Vertex 8: 4
Vertex 9: 4
Vertex 10: 4
Vertex 11: 5
Vertex 12: 5
Vertex 13: 6
Vertex 14: -1
```
