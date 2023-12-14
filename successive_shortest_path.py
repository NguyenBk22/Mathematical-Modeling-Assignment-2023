import networkx as nx
import random

def successive_shortest_path(graph, start=0):
    def bellman_ford(start, n, flow, graph):
        distance = [float('inf')] * n
        distance[start] = 0
        parent = [-1] * n
        for _ in range(n):
            for u in range(n):
                for v in range(n):
                    edge_data = graph.get_edge_data(u, v)
                    if edge_data is not None:
                        if flow[u][v] < edge_data['capacity']:
                            if distance[u] + edge_data['cost'] < distance[v]:
                                distance[v] = distance[u] + edge_data['cost']
                                parent[v] = u

        return distance, parent

    n = len(graph.nodes)
    destination = n - 1
    flow = [[0] * n for _ in range(n)]
    while True:
        distance, parent = bellman_ford(start, n, flow, graph)
        if distance[destination] == float('inf'):
            break

        delta = float('inf')
        v = destination

        while v != start:
            u = parent[v]
            delta = min(delta, graph[u][v]['capacity'] - flow[u][v])
            v = u

        v = destination

        while v != start:
            u = parent[v]
            flow[u][v] += delta
            flow[v][u] -= delta
            v = u

    # Update the graph with flow information
    for u, v, data in graph.edges(data=True):
        data['flow'] = flow[u][v]

    min_cost = sum(flow[u][v] * data['cost']
                   for u, v, data in graph.edges(data=True))

    return min_cost, graph
