#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void bfsKernel(
    int numVertices,
    const int *adjStart,
    const int *adjList,
    int *dist,
    int currentLevel
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex >= numVertices)
        return;

    // Process only vertices discovered at the current level
    if (dist[vertex] != currentLevel)
        return;

    // Visit all neighbours of this vertex
    for (int edge = adjStart[vertex];
         edge < adjStart[vertex + 1];
         edge++) {

        int neighbour = adjList[edge];

        // If neighbour has not been visited
        if (dist[neighbour] == -1) {
            dist[neighbour] = currentLevel + 1;
        }
    }
}

int main() {

    const int numVertices = 15;
    const int numEdges = 38;

    // CSR offsets
    int hostOffsets[numVertices + 1] =
    {
         0,   // vertex 0
         2,   // vertex 1
         5,   // vertex 2
         7,   // vertex 3
        10,   // vertex 4
        13,   // vertex 5
        15,   // vertex 6
        18,   // vertex 7
        21,   // vertex 8
        23,   // vertex 9
        26,   // vertex 10
        29,   // vertex 11
        31,   // vertex 12
        34,   // vertex 13
        36,   // vertex 14
        38    // end
    };

    // CSR adjacency list
    int hostEdges[numEdges] =
    {

        1, 2,               // Vertex 0
        0, 3, 4,            // Vertex 1
        0, 4,               // Vertex 2
        1, 6, 7,            // Vertex 3
        1, 2, 5,            // Vertex 4
        4, 8,               // Vertex 5
        3, 9, 10,           // Vertex 6
        3, 8, 10,           // Vertex 7
        5, 7, 11,           // Vertex 8
        6, 12,              // Vertex 9
        6, 7, 11,           // Vertex 10
        8, 10,              // Vertex 11 
        9, 13,              // Vertex 12
        9, 12,              // Vertex 13
        12, 13              // Vertex 14 
    };

    int sourceVertex = 0;

    int *deviceOffsets;
    int *deviceEdges;
    int *deviceDistances;

    cudaMalloc(&deviceOffsets,(numVertices + 1) * sizeof(int));

    cudaMalloc(&deviceEdges,numEdges * sizeof(int));

    cudaMalloc(&deviceDistances,numVertices * sizeof(int));

    cudaMemcpy(deviceOffsets,hostOffsets,(numVertices + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(deviceEdges,hostEdges,numEdges * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemset(deviceDistances,-1,numVertices * sizeof(int));

    int initialDistance = 0;

    cudaMemcpy(&deviceDistances[sourceVertex],&initialDistance,sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int numberOfBlocks =
        (numVertices + threadsPerBlock - 1)
        / threadsPerBlock;

    for (int currentLevel = 0;
         currentLevel < numVertices;
         currentLevel++) {

        bfsKernel<<<numberOfBlocks, threadsPerBlock>>>(numVertices,deviceOffsets,deviceEdges,deviceDistances,currentLevel);

        cudaDeviceSynchronize();
    }

    vector<int> hostDistances(numVertices);

    cudaMemcpy(hostDistances.data(),deviceDistances,numVertices * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "Shortest distances from vertex "
         << sourceVertex << ":\n\n";

    for (int vertex = 0;
         vertex < numVertices;
         vertex++) {

        cout << "Vertex "
             << vertex
             << ": "
             << hostDistances[vertex]
             << '\n';
    }

    cudaFree(deviceOffsets);
    cudaFree(deviceEdges);
    cudaFree(deviceDistances);

    return 0;
}