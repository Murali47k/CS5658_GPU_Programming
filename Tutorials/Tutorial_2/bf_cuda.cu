#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;

__global__ void bellmanFord(int V,const int *nodes,const int *edges,int *dist,bool *visited){
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    if (dist[u] == INT_MAX)
        return;

    int start = nodes[u];
    int end   = nodes[u + 1];

    for (int i = start; i < end; i++){

        int v      = edges[2 * i];
        int weight = edges[2 * i + 1];

        int newDistance = dist[u] + weight;

        if (newDistance < dist[v]){
            atomicMin(&dist[v], newDistance);
            *visited = true;
        }
    }
}

int main(){

    const int V = 10;

    int h_nodes[V + 1] = {0, 3, 5, 8, 10, 12, 15, 17, 19, 20, 20};

    int h_edges[] = {1,4,2,2,3,9,2,3,4,6,3,1,5,7,4,5,4,2,6,5,5,3,7,8,6,1,8,6,9,10,7,2,9,9,8,3,9,4,9,1};

    const int E = 20;

    vector<int> h_dist(V, INT_MAX);

    int s = 0;
    h_dist[s] = 0;

    int *d_nodes;
    int *d_edges;
    int *d_dist;
    bool *d_visited;

    cudaMalloc(&d_nodes, (V + 1) * sizeof(int));
    cudaMalloc(&d_edges, 2 * E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_visited, sizeof(bool));

    cudaMemcpy(d_nodes,h_nodes,(V + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_edges,h_edges,2 * E * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_dist,h_dist.data(),V * sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (V + threadsPerBlock - 1) / threadsPerBlock;

    clock_t start_time = clock();

    for (int i = 0; i < V - 1; i++)
    {
        bool h_visited = false;

        cudaMemcpy(d_visited,&h_visited,sizeof(bool),cudaMemcpyHostToDevice);

        bellmanFord<<<blocks, threadsPerBlock>>>(V,d_nodes,d_edges,d_dist,d_visited);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_visited,d_visited,sizeof(bool),cudaMemcpyDeviceToHost);

        if (!h_visited)
            break;
    }

    clock_t end_time = clock();
    double elapsed_ms = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    cudaMemcpy(h_dist.data(),d_dist,V * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "Shortest distances from vertex "
         << s << ":\n";

    for (int v = 0; v < V; v++){
        cout << "Vertex " << v << ": ";

        if (h_dist[v] == INT_MAX)
            cout << "INT_MAX";
        else
            cout << h_dist[v];

        cout << '\n';
    }

    cout << "Execution time: " << elapsed_ms << " ms\n";

    cudaFree(d_nodes);
    cudaFree(d_edges);
    cudaFree(d_dist);
    cudaFree(d_visited);

    return 0;
}
