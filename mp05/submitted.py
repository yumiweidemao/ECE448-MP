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

    return []
