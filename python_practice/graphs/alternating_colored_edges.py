# https://leetcode.com/problems/shortest-path-with-alternating-colors/

from collections import defaultdict, deque


class Solution:

    def shortestAlternatingPaths(self, n, redEdges, blueEdges):
        # Create adjacency lists for red and blue edges
        red_graph = defaultdict(list)
        blue_graph = defaultdict(list)
        for u, v in redEdges:
            red_graph[u].append(v)
        for u, v in blueEdges:
            blue_graph[u].append(v)

        # Initialize the distances array with -1 (unreachable)
        distances = [-1] * n
        distances[0] = 0  # The distance from the start node to itself is always 0

        # Initialize the queue and visited set
        # Format in queue: (node, steps, lastColorUsed), where lastColorUsed is 'R', 'B', or None
        queue = deque([(0, 0, 'R'), (0, 0, 'B')])
        visited = set([(0, 'R'), (0, 'B')])

        while queue:
            node, steps, last_color = queue.popleft()

            # Get the next color to use and the corresponding graph
            next_colors = [('B', blue_graph)] if last_color == 'R' else [('R', red_graph)]

            for color, graph in next_colors:
                for neighbor in graph[node]:
                    if (neighbor, color) not in visited:
                        visited.add((neighbor, color))
                        queue.append((neighbor, steps + 1, color))
                        # Update the distance for the neighbor if it's the first time reaching it,
                        # or if we found a shorter path
                        # NOTE: distances[neighbor] > steps + 1 is not required as we are traversing level by level.
                        if distances[neighbor] == -1 or distances[neighbor] > steps + 1:
                            distances[neighbor] = steps + 1

        return distances


    def shortestAlternatingPaths_veera(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        '''
            Previous Node: (answer[i-1], previous_edge_color)

            Current Node:
                if answer[i-1] == -1:
                    store the tuple as (-1, current edge color)
                else:
                    if (the previous edge color and current edge color are different) or (current_edge_color and self edge of previous node color are different)
                        answer[i] = answer[i-1] + 1
                        current_edge_color = current edge color
                    else:
                        answer[i] = -1
                        current_edge_color = current edge color
            BFS starting from 0 node
        '''

        blue_graph = defaultdict(set)
        for x, y in blueEdges:
            blue_graph[x].add(y)

        red_graph = defaultdict(set)
        for x, y in redEdges:
            red_graph[x].add(y)

        result = [-1] * n # Assuming all the nodes are not reachable
        result[0] = 0 # First node result is 0


        queue = deque([(0, 0, None)]) # We don't know the color.
        seen = set()

        while queue:
            node, distance, color = queue.popleft()

            if color is None:
                queue.append((0, 0, 0))
                queue.append((0, 0, 1))
                continue

            next_color = 1 - color  # Switch color: 0 -> 1 or 1 -> 0
            next_graph = blue_graph[node] if not color else red_graph[node]

            for neighbor in next_graph:
                if (neighbor, next_color) not in seen:
                    seen.add((neighbor, next_color))
                    if result[neighbor] == -1: # or result[neighbor] > distance + 1
                        # Or case will never happen as we are traversing level by level.
                        result[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1, next_color))

        return result
