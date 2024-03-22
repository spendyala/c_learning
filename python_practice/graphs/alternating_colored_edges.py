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
                        if distances[neighbor] == -1 or distances[neighbor] > steps + 1:
                            distances[neighbor] = steps + 1

        return distances
