import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import threading
import tracemalloc


class comparison:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.graph = self.generate_random_grid_graph()

    def generate_random_grid_graph(self, capacity_range=(1, 10), cost_range=(1, 5)):
        grid_graph = nx.DiGraph()

        # Add nodes to the graph
        nodes = [i * self.cols +
                 j for i in range(self.rows) for j in range(self.cols)]
        grid_graph.add_nodes_from(nodes)

        for i in nodes:
            if i % self.cols < self.cols - 1:
                grid_graph.add_edge(
                    i, i + 1, capacity=random.randint(*capacity_range), cost=random.randint(*cost_range))

            # Add downward edge
            if i // self.cols < self.rows - 1:
                grid_graph.add_edge(
                    i, i + self.cols, capacity=random.randint(*capacity_range), cost=random.randint(*cost_range))

        return grid_graph

    def visualize_grid_graph(self):
        # Generate a random directed grid graph
        # Visualize the directed grid graph
        # Adjust layout for better visualization
        pos = {node: ((node % self.cols), -(node // self.cols))
               for node in self.graph.nodes()}
        options = {
            "font_size": 8,
            "node_size": 200,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
        nx.draw(self.graph, pos, with_labels=True, **options)
        edge_labels = {(u, v): f"({d['capacity']},{d['cost']})" for (
            u, v, d) in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=6)
        # Show the plot
        plt.show()

    def reshape(self, tuple: (int, int)):
        self.rows = tuple[0]
        self.cols = tuple[1]
        self.graph = self.generate_random_grid_graph()

    def compare_time_elapsed(self, func1, func2):
        graph1 = self.graph.copy()
        graph2 = self.graph.copy()

        def run_func1(result1):
            nonlocal graph1
            try:
                start_time = time.time()
                func1(graph1)
                result1.append(time.time() - start_time)
            except Exception as e:
                print(f"Exception in func1: {e}")
                result1.append(None)

        def run_func2(result2):
            nonlocal graph2
            try:
                start_time = time.time()
                func2(graph2)
                result2.append(time.time() - start_time)
            except Exception as e:
                print(f"Exception in func2: {e}")
                result2.append(None)

        # Create threads
        result1 = []
        result2 = []
        threads = [threading.Thread(target=run_func1, args=(result1,)),
                threading.Thread(target=run_func2, args=(result2,))]

        # Start the threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return result1[0], result2[0]

    def compare_memory_usage(self, func1, func2):
        graph1 = self.graph.copy()
        graph2 = self.graph.copy()

        def run_func1(result1):
            nonlocal graph1
            tracemalloc.clear_traces()
            tracemalloc.start()  # Start tracing memory allocations
            func1(graph1)
            # Return the peak memory usage
            result1.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.reset_peak()  # Stop tracing memory allocations

        def run_func2(result2):
            nonlocal graph2
            tracemalloc.clear_traces()
            tracemalloc.start()
            func2(graph2)
            # Return the peak memory usage
            result2.append(tracemalloc.get_traced_memory()[1])
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
            self.graph = self.generate_random_grid_graph()
            print(
                f"---------------------------Epoch {i+1}------------------------------")
            t1, t2 = self.compare_time_elapsed(alt1, alt2)
            m1, m2 = self.compare_memory_usage(alt1, alt2)
            print(
                f"""                     Algorithm 1                    Algorithm 2
    Time Elapsed:   {t1 * 1000:.2f} ms            {t2 * 1000:.2f} ms
    Memory Usage:   {m1 / 1024:.2f} KB            {m2 / 1024:.2f} KB
----------------------------------------------------------------""")
