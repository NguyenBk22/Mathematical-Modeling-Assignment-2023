import numpy as np
import networkx as nx

def cycle_cancelling(graph, cost, initial_flow):
    n = graph.number_of_nodes()
    flow = np.array(initial_flow)

    def find_negative_cycle():
        dist = [float('inf')] * n
        parent = [-1] * n
        source = 0
        dist[source] = 0
        for _ in range(n):
            for u, v in graph.edges():
                if graph[u][v]['capacity'] > 0 and dist[u] + cost[u][v] < dist[v]:
                    dist[v] = dist[u] + cost[u][v]
                    parent[v] = u

        for u, v in graph.edges():
            if graph[u][v]['capacity'] > 0 and dist[u] + cost[u][v] < dist[v]:
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

#Hàm thêm cạnh vào đồ thị từ ma trận dung lượng
def add_edges_from_matrix_capacity(graph, capacity_matrix):
    num_nodes = capacity_matrix.shape[0]
    for u in range(num_nodes):
        for v in range(num_nodes):
            capacity = capacity_matrix[u][v]
            if capacity > 0:
                graph.add_edge(u, v, capacity=capacity)

# Tạo đồ thị mạng
graph = nx.DiGraph()

# Thêm cạnh và chi phí
#graph.add_edge(0, 1, capacity=4)
#graph.add_edge(0, 2, capacity=2)
#graph.add_edge(1, 2, capacity=2)
#graph.add_edge(1, 3, capacity=3)
#graph.add_edge(2, 3, capacity=5)

# Tạo ma trận dung lượng
capacity_matrix = np.array(
    [
        [0, 4, 2, 0],
        [0, 0, 2, 3],
        [0, 0, 0, 5],
        [0, 0, 0, 0]
    ]
)
# Thêm cạnh vào đồ thị từ ma trận dung lượng
add_edges_from_matrix_capacity(graph, capacity_matrix)

# Tạo ma trận chi phí
cost = np.array([
    [0, 2, 2, 0],
    [0, 0, 1, 3],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

# Tạo ma trận luồng ban đầu
initial_flow = np.array([
    [0, 3, 1, 0],
    [0, 0, 0, 3],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

# Gọi hàm cycle_cancelling để giải bài toán Min-Cost Flow
final_flow = cycle_cancelling(graph, cost, initial_flow)

# In kết quả
print("Final flow:")
print(final_flow)

# Tính chi phí
min_cost = 0
for u, v in graph.edges():
    min_cost += final_flow[u][v] * cost[u][v]
print("Min cost:", min_cost)
