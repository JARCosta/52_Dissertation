import numpy as np

class BFS:

    def __init__(self, graph_matrix):
        self.graph_matrix = np.maximum(graph_matrix, graph_matrix.T)

        self.to_visit = set(np.random.permutation(range(graph_matrix.shape[0])))
        
        self.visited = set()
        self.visiting = set()

    
    def visit(self, node):
        # print(f"Visiting {node}, {len(self.to_visit)} nodes left")
        self.visited.add(node)
        
        # print(f"Visiting {node}, {np.where(self.graph_matrix[node] != 0)[0]}")
        for neighbor in np.where(self.graph_matrix[node] != 0)[0]:
            # print(f"Neighbor {neighbor}: to_visit: {neighbor in self.to_visit}, visited: {neighbor in self.visited}, visiting: {neighbor in self.visiting}")
            if neighbor in self.to_visit and neighbor not in self.visited and neighbor not in self.visiting:
                self.visiting.add(neighbor)
                self.to_visit.remove(neighbor)

    def connected_component(self):
        self.visited = set()
        self.visiting = set()

        # start with a random node
        random_start = np.random.choice(list(self.to_visit))
        self.visiting.add(random_start)
        self.to_visit.remove(random_start)

        while self.visiting:
            self.visit(self.visiting.pop())
        # print(f"Connected component of size {len(self.visited)}")
        # if len(self.visited) == 1:
        #     print("Only one node in the connected component")
        #     breakpoint()
        return self.visited


    def connected(self):

        components = []
        while self.to_visit:
            components.append(np.array(list(self.connected_component())).astype(int))

        # print([len(c) for c in components])
        # print(len(components))
        return components