import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import threading
import tracemalloc


class comparison:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.graph = None

    def generate_random_grid_graph(self, rows, cols, capacity_range=(1, 10), cost_range=(1, 5)):
        grid_graph = nx.DiGraph()

        # Add nodes to the graph
        nodes = [i * cols + j for i in range(rows) for j in range(cols)]
        grid_graph.add_nodes_from(nodes)

        for i in nodes:
            if i % cols < cols - 1:
                grid_graph.add_edge(
                    i, i + 1, capacity=random.randint(*capacity_range), cost=random.randint(*cost_range))

            # Add downward edge
            if i // cols < rows - 1:
                grid_graph.add_edge(
                    i, i + cols, capacity=random.randint(*capacity_range), cost=random.randint(*cost_range))

        return grid_graph

    def visualize_grid_graph(self, rows, cols):
        # Generate a random directed grid graph
        random_directed_grid_graph = generate_random_grid_graph(rows, cols)

        # Visualize the directed grid graph
        # Adjust layout for better visualization
        pos = {node: ((node % cols), -(node // cols))
               for node in random_directed_grid_graph.nodes()}
        options = {
            "font_size": 8,
            "node_size": 200,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
        nx.draw(random_directed_grid_graph, pos, with_labels=True, **options)
        edge_labels = {(u, v): f"({d['capacity']},{d['cost']})" for (
            u, v, d) in random_directed_grid_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            random_directed_grid_graph, pos, edge_labels=edge_labels, font_size=6)
        # Show the plot
        plt.show()

    def compare_time_elapsed(self, func1, func2):
        # Create a copy of the graph for each function to ensure fairness in comparison
        graph1 = self.graph.copy()
        graph2 = self.graph.copy()

        def run_func1(result):
            nonlocal graph1
            start_time = time.time()
            func1(graph1)
            result.append(time.time() - start_time)

        def run_func2(result):
            nonlocal graph2
            start_time = time.time()
            func2(graph2)
            result.append(time.time() - start_time)

        # Create threads
        results = []
        threads = [threading.Thread(target=run_func1, args=(
            results,)), threading.Thread(target=run_func2, args=(results,))]

        # Start the threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return results[0], results[1]

    def compare_memory_usage(self, func1, func2):
        graph1 = self.graph.copy()
        graph2 = self.graph.copy()

        def run_func1(result1):
            nonlocal graph1
            tracemalloc.clear_traces()
            tracemalloc.start()  # Start tracing memory allocations
            func1(graph1)
            result1.append(tracemalloc.get_traced_memory()[1])  # Return the peak memory usage
            tracemalloc.reset_peak()  # Stop tracing memory allocations

        def run_func2(result2):
            nonlocal graph2
            tracemalloc.clear_traces()
            tracemalloc.start()
            func2(graph2)
            result2.append(tracemalloc.get_traced_memory()[1])  # Return the peak memory usage
            tracemalloc.reset_peak()

        result1 = []
        result2 = []
        threads = [threading.Thread(target=run_func1, args=(
            result1,)), threading.Thread(target=run_func2, args=(result2,))]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        return result1[0], result2[0]

    def compare_2_algorithms(self, alt1, alt2, epoch=10):
        for i in range(epoch):
            self.graph = self.generate_random_grid_graph(
                self.rows, self.columns)
            print(
                f"---------------------------Epoch {i+1}------------------------------")
            t1, t2 = self.compare_time_elapsed(alt1, alt2)
            m1, m2 = self.compare_memory_usage(alt1, alt2)
            print(
                f"""                     Algorithm 1                    Algorithm 2
    Time Elapsed:   {t1 * 1000:.2f} ms            {t2 * 1000:.2f} ms
    Memory Usage:   {m1 / 1024:.2f} KB            {m2 / 1024:.2f} KB
----------------------------------------------------------------""")



