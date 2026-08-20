#include <bits/stdc++.h>
#include <time.h>

using namespace std;

int main()
{
    const int V = 10;

    int nodes[V + 1] = {0, 3, 5, 8, 10, 12, 15, 17, 19, 20, 20};

    int edges[] = {1,4,2,2,3,9,2,3,4,6,3,1,5,7,4,5,4,2,6,5,5,3,7,8,6,1,8,6,9,10,7,2,9,9,8,3,9,4,9,1};

    const int E = 20;

    vector<int> dist(V, INT_MAX);

    int s = 0;
    dist[s] = 0;

    clock_t start_time = clock();

    for (int i = 0; i < V - 1; i++){

        bool visited = false;

        for (int u = 0; u < V; u++){

            if (dist[u] == INT_MAX)
                continue;

            int start = nodes[u];
            int end   = nodes[u + 1];

            for (int i = start; i < end; i++){

                int v      = edges[2 * i];
                int weight = edges[2 * i + 1];

                if (dist[u] + weight < dist[v]){
                    dist[v] = dist[u] + weight;
                    visited = true;
                }
            }
        }

        if (!visited)
            break;
    }

    clock_t end_time = clock();
    double elapsed_ms = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

    cout << "Shortest distances from vertex " << s << ":\n";

    for (int v = 0; v < V; v++) {
        cout << "Vertex " << v << ": ";
        if (dist[v] == INT_MAX)
            cout << "INT_MAX";
        else
            cout << dist[v];
        cout << '\n';
    }

    cout << "Execution time: " << elapsed_ms << " ms\n";

    return 0;
}
