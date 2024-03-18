from collections import defaultdict
from typing import List
class Solution:
    def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:

        restricted_set = set(restricted)

        graph = defaultdict(list)

        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        seen = set()
        stack = []

        seen.add(0)
        stack.append(0)
        reached = 1

        while stack:
            node = stack.pop()

            neighbors = graph[node]

            for neighbor in neighbors:
                if neighbor in restricted_set:
                    continue
                if neighbor not in seen:
                    reached += 1
                    seen.add(neighbor)
                    stack.append(neighbor)
        return reached