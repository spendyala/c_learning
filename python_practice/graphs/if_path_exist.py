# https://leetcode.com/problems/find-if-path-exists-in-graph/

class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:

        if source == destination:
            return True

        graph = {}
        for x, y in edges:
            graph.setdefault(x, [])
            graph.setdefault(y, [])
            graph[x].append(y)
            graph[y].append(x)

        visited = {source}
        stack = []
        stack.append(source)
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor == destination:
                    return True
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited.add(neighbor)

        return False





