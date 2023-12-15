import numpy as np
import networkx as nx
import random

def cycle_cancelling(graph):
    def randomized_initial_flow(graph, source, sink):
        num_nodes = graph.number_of_nodes()
        initial_flow = np.zeros((num_nodes, num_nodes))

        edges = list(graph.edges())
        # print("edges11:", edges)
        while True:
            random.shuffle(edges)
            feasible_path = find_feasible_path(graph, source, sink, edges)
            # print("edge2",edges)
            if feasible_path is None:
                break

            min_capacity = float('inf')
            for i in range(len(feasible_path) - 1):
                u = feasible_path[i]
                v = feasible_path[i + 1]
                min_capacity = min(min_capacity, graph[u][v]['capacity'])

            for i in range(len(feasible_path) - 1):
                u = feasible_path[i]
                v = feasible_path[i + 1]
                initial_flow[u][v] += min_capacity

                # Giảm dung lượng cạnh để cập nhật đồ thị
                graph[u][v]['capacity'] -= min_capacity

        return initial_flow

    def find_feasible_path(graph, source, sink, edges):
        visited = set()
        stack = [(source, [])]

        while stack:
            current_node, path = stack.pop()

            if current_node == sink:
                return path + [sink]

            visited.add(current_node)

            for edge in edges:
                u, v = edge
                if u == current_node and v not in visited and graph[u][v]['capacity'] > 0:
                    stack.append((v, path + [u]))
        return None

    def sub_cycle_cancelling(graph, initial_flow):
        n = graph.number_of_nodes()
        flow = np.array(initial_flow)

        def find_negative_cycle():
            dist = [float('inf')] * n
            parent = [-1] * n
            source = 0
            dist[source] = 0
            for _ in range(n):
                for u, v in graph.edges():
                    if graph[u][v]['capacity'] > 0 and dist[u] + graph[u][v]['cost'] < dist[v]:
                        dist[v] = dist[u] + graph[u][v]['cost']
                        parent[v] = u

            for u, v in graph.edges():
                if graph[u][v]['capacity'] > 0 and dist[u] + graph[u][v]['cost'] < dist[v]:
                    cycle = [v]
                    while cycle[0] != v or len(cycle) <= 1:
                        cycle.insert(0, parent[cycle[0]])
                    return cycle

            return None

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

        return flow

    source = 0
    sink = int(len(graph.nodes)) - 1
    list_initial_flow = []

    def initial_flow_not_in_list(initial_flow, list_initial_flow):
        for i in range(len(list_initial_flow)):
            if np.array_equal(initial_flow, list_initial_flow[i]):
                return False
        return True

    rg = 10*sink

    for i in range(rg):
        graph_temp = graph.copy()
        initial_flow = randomized_initial_flow(graph_temp, source, sink)
        #nếu initial flow không có trong list thì thêm vào
        if initial_flow_not_in_list(initial_flow, list_initial_flow):
            list_initial_flow.append(initial_flow)

    min_cost = float('inf')
    final_flow = np.zeros((len(graph.nodes),len(graph.nodes)))
    for i in range(len(list_initial_flow)):
        graph_temp = graph.copy()
        final_flow_tp = sub_cycle_cancelling(graph_temp, list_initial_flow[i])
        minc = 0
        for u, v in graph.edges():
            minc += final_flow_tp[u][v] * graph[u][v]['cost']
        if minc < min_cost:
            min_cost = minc
            final_flow = final_flow_tp

        for u, v, data in graph.edges(data=True):
            data['flow'] = int(final_flow[u][v])
    return min_cost, graph

