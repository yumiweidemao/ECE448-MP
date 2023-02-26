# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    if len(maze.waypoints) > 1:
        print("BFS can only run on single waypoint")
        return []

    path = []
    parent = dict()
    visited = set()
    q = queue.Queue()
    q.put(maze.start)
    visited.add(maze.start)
    parent[maze.start] = None
    while not q.empty():
        current = q.get()
        if current == maze.waypoints[0]:
            break
        for neighbor in maze.neighbors(current[0], current[1]):
            if neighbor not in visited:
                parent[neighbor] = current
                q.put(neighbor)
                visited.add(neighbor)

    x = maze.waypoints[0]
    while x is not None:
        path.append(x)
        x = parent[x]

    path.reverse()
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    pq = queue.PriorityQueue()
    path = []
    b = maze.waypoints[0]
    h = lambda a : abs(a[0] - b[0]) + abs(a[1] - b[1])
    visited = set()
    parent = dict()
    start = (h(maze.start), maze.start, 0)  # (h, indices, g)
    parent[maze.start] = None
    visited.add(maze.start)
    pq.put(start)
    while not pq.empty():
        curr = pq.get()
        if curr[1] == b:
            break
        for neighbor in maze.neighbors(curr[1][0], curr[1][1]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = curr[1]
                pq.put((h(neighbor)+curr[2]+1, neighbor, curr[2]+1))

    while b is not None:
        path.append(b)
        b = parent[b]

    path.reverse()
    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    path = [maze.start]
    waypoints = list(maze.waypoints)
    pq = queue.PriorityQueue()
    parent = dict()
    parent[maze.start] = None
    visited = set()

    h = lambda a : (nearest_waypoint_dist(a, waypoints) + MST_length(waypoints, a)) # heuristic

    start = (h(maze.start), maze.start, 0)
    pq.put(start)
    visited.add(maze.start)

    while not pq.empty():
        curr = pq.get()
        if curr[1] in waypoints:
            waypoints.remove(curr[1])
            visited = set()
            visited.add(curr[1])
            subpath = []
            x = curr[1]
            while x is not None:
                subpath.append(x)
                x = parent[x]
            subpath.reverse()
            path += subpath[1:]
            parent = dict()
            parent[curr[1]] = None
            pq = queue.PriorityQueue()
        if not waypoints:
            break
        for neighbor in maze.neighbors(curr[1][0], curr[1][1]):
            if neighbor in visited:
                continue
            parent[neighbor] = curr[1]
            pq.put((h(neighbor)+curr[2]+1, neighbor, curr[2]+1))
            visited.add(neighbor)

    return path


def md(a, b):
    # manhattan distance between a and b
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def nearest_waypoint_dist(current, waypoints):
    # return the nearest waypoint distance
    min_d = 999
    for i in range(len(waypoints)):
        d = md(current, waypoints[i])
        if d < min_d:
            min_d = d
    return min_d


def waypoints2graph(waypoints):
    # return a graph represented by a list of edges, each edge is a tuple
    # (length, (a, b))
    edges = []
    for i in range(len(waypoints)):
        for j in range(i+1, len(waypoints)):
            edges.append((md(waypoints[i], waypoints[j]), (i, j)))
    return edges


"""
This doesn't work because Kruskal's algorithm cannot generate a MST given a starting vertex.

def MST_length(waypoints):
    # return the length of the MST of a graph using Kruskal's algorithm
    # return -1 if no MST possible
    class DisjointSet:
        def __init__(self, n):
            self.parent = list(range(n))

        def find(self, x):
            if x == self.parent[x]:
                return x
            return self.find(self.parent[x])

        def union(self, x, y):
            self.parent[x] = y

        def connected(self):
            x = self.find(0)
            for i in range(1, len(self.parent)):
                if self.find(i) != x:
                    return False
            return True

    ds = DisjointSet(len(waypoints))
    edges = waypoints2graph(waypoints)
    pq = queue.PriorityQueue()
    length = 0

    for edge in edges:
        pq.put(edge)
    while not pq.empty():
        curr = pq.get()
        a, b = curr[1]
        if ds.find(a) != ds.find(b):
            ds.union(a, b)
            length += curr[0]
        if ds.connected():
            return length

    return -1
"""
def MST_length(waypoints, a):
    # return the length of the minimum spanning tree given a.

    # find the nearest vertices of a in waypoints
    min_d = 999
    vertices = []
    for i in range(len(waypoints)):
        d = md(a, waypoints[i])
        if d < min_d:
            min_d = d
            vertices = [i]
        elif d == min_d:
            vertices.append(i)

    edges = waypoints2graph(waypoints)
    min_weight = min_spanning_tree_weight(edges, vertices)

    return min_weight

def min_spanning_tree_weight(edges, s):
    if not edges:
        return 0

    adj_list = {}
    for weight, (a, b) in edges:
        if a not in adj_list:
            adj_list[a] = []
        if b not in adj_list:
            adj_list[b] = []
        adj_list[a].append((b, weight))
        adj_list[b].append((a, weight))

    min_weight = float('inf')

    for start_vertex in s:
        visited = {start_vertex}
        pq = queue.PriorityQueue()
        for neighbor, neighbor_weight in adj_list[start_vertex]:
            pq.put((neighbor_weight, neighbor))
        while not pq.empty():
            weight, vertex = pq.get()
            if vertex in visited:
                continue
            visited.add(vertex)
            if len(visited) == len(adj_list):
                tree_weight = sum(edge_weight for _, edge_weight in pq.queue)
                min_weight = min(min_weight, tree_weight)
                break
            for neighbor, neighbor_weight in adj_list[vertex]:
                if neighbor not in visited:
                    pq.put((neighbor_weight, neighbor))

    return min_weight

def main():
    # add some tests here
    waypoints = ((1, 1), (2, 2), (3, 3), (2, 5), (3, 6))
    print("MST length =", MST_length(waypoints, (1, 2)))

if __name__ == "__main__":
    main()