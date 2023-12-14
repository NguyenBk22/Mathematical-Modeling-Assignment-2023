import numpy as np
import networkx as nx

def cycle_cancelling(graph, initial_flow):
    n = graph.number_of_nodes()
    flow = np.array(initial_flow)

    def find_negative_cycle():
        dist = [float('inf')] * n
        parent = [-1] * n
        source = 0
        dist[source] = 0
        for _ in range(n):
            for u, v in graph.edges():
                if graph[u][v]['capacity'] > 0 and dist[u] + graph.graph['cost'] < dist[v]:
                    dist[v] = dist[u] + graph.graph['cost']
                    parent[v] = u

        for u, v in graph.edges():
            if graph[u][v]['capacity'] > 0 and dist[u] + graph.graph['cost'] < dist[v]:
                cycle = [v]
                while cycle[0] != v or len(cycle) <= 1:
                    cycle.insert(0, parent[cycle[0]])
                return cycle

        return None

    min_cost = float('inf')

    while True:
        negative_cycle = find_negative_cycle()
        if negative_cycle is None:
            break
        delta = min([graph[u][v]['capacity'] for u, v in zip(negative_cycle, negative_cycle[1:])])
        for u, v in zip(negative_cycle, negative_cycle[1:]):
            flow[u][v] += delta
            flow[v][u] -= delta
            graph[u][v]['capacity'] -= delta
            graph[v][u]['capacity'] += delta

        # Update the 'flow' attribute of each edge
        for u, v in graph.edges():
            graph[u][v]['flow'] = flow[u][v]

        # Update the minimum cost
        min_cost = min(min_cost, delta * graph.graph['cost'])

    return graph, min_cost
