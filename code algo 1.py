


def bellman_ford(start, n, flow, capacity, costo):
    distance = [float('inf')] * n
    distance[start] = 0
    parent = [-1] * n
    for _ in range(n):
        for u in range(n):
            for v in range(n):
                if flow[u][v] < capacity[u][v]:
                    if distance[u] + costo[u][v] < distance[v]:
                        distance[v] = distance[u] + costo[u][v]
                        parent[v] = u

    return distance, parent
# distance dùng để tìm chi phí nhỏ nhất ( tính theo costo ), parent mảng chứa đỉnh cha của các đỉnh


def successive_shortest_path(capacity, start, destination, costo):
    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    while True:
        # sau mỗi lần lặp sẽ cập nhận flow
        # flow là giá trị các ống mà luồng thứu nhất đi qua
        distance, parent = bellman_ford(start, n, flow, capacity, costo)
        if distance[destination] == float('inf'):
            break

        delta = float('inf')
        v = destination
        # duyệt đường đi từ cuối về đầu
        # lường cực đại áp dụng cho delta, delta = giá trị tối đa có thể đi qua luồng
        while v != start:
            u = parent[v]
            delta = min(delta, capacity[u][v] - flow[u][v])
            v = u

        v = destination

        while v != start:
            u = parent[v]
            flow[u][v] += delta
            flow[v][u] -= delta
            v = u
    min_costo = 0
    for u in range(n):
        for v in range(n):
            min_costo += flow[u][v] * costo[u][v]

    return min_costo, flow


# Esempio sử dụng
capacity = [
    [0, 4, 2, 0],
    [0, 0, 2, 3],
    [0, 0, 0, 5],
    [0, 0, 0, 0]
]

costo = [
    [0, 2, 2, 0],
    [0, 0, 1, 3],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
]

start = 0
destination = 3

min_costo, flow = successive_shortest_path(capacity, start, destination, costo)
print("Costo minimo:", min_costo)
print("flow finale:")
for row in flow:
    print(row)
