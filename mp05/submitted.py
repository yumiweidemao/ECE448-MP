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
    path = []
    pq = queue.PriorityQueue()
    parent = dict()
    parent[(maze.start, maze.waypoints, 0)] = None
    visited = dict()
    x = None

    def h(a, waypoints):
        return nearest_waypoint_dist(a, waypoints) + MST_length(waypoints)

    start = (h(maze.start, maze.waypoints), maze.start, 0, maze.waypoints)
    pq.put(start)

    while not pq.empty():
        curr = pq.get()
        waypoints = curr[3]
        if not waypoints:
            x = (curr[1], waypoints, curr[2])
            break
        if (curr[1], waypoints) in visited and visited[(curr[1], waypoints)] <= h(curr[1], waypoints)+curr[2]+1:
            continue
        visited[(curr[1], waypoints)] = h(curr[1], waypoints) + curr[2] + 1
        for neighbor in maze.neighbors(curr[1][0], curr[1][1]):
            new_waypoints = waypoints[:]
            if neighbor in new_waypoints:
                new_waypoints = list(new_waypoints)
                new_waypoints.remove(neighbor)
                new_waypoints = tuple(new_waypoints)
            parent[(neighbor, new_waypoints, curr[2]+1)] = (curr[1], waypoints, curr[2])
            pq.put((h(neighbor, new_waypoints)+curr[2]+1, neighbor, curr[2]+1, new_waypoints))

    while x is not None:
        path.append(x[0])
        x = parent[x]

    path.reverse()
    return path


def md(a, b):
    # manhattan distance between a and b
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def nearest_waypoint_dist(current, waypoints):
    # return the nearest waypoint distance
    if not waypoints:
        return 0
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
            self.parent[self.find(x)] = self.find(y)

        def connected(self):
            x = self.find(0)
            for i in range(1, len(self.parent)):
                if self.find(i) != x:
                    return False
            return True

    if not waypoints:
        return 0

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

    return 0

def main():
    # add some tests here
    waypoints = ((1, 6), (1, 10), (1, 14), (1, 18), (1, 22), (1, 26), (5, 4))
    print("MST length =", MST_length(waypoints))

if __name__ == "__main__":
    main()