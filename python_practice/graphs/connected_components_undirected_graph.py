# https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:

        ans = 0

        graph = {}
        for x, y in edges:
            graph.setdefault(x, [])
            graph.setdefault(y, [])
            graph[x].append(y)
            graph[y].append(x)

        seen = set()

        for i in range(n):
            stack = [i]
            if i not in seen:
                ans += 1
                seen.add(i)
            else:
                continue
            if i not in graph:
                continue


            while stack:
                node = stack.pop()
                for neighbor in graph[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

        return ans