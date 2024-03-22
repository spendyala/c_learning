# https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/description/

from collections import deque
from typing import List
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:

        nrows = len(grid)
        ncols = len(grid[0])

        if k >= nrows + ncols - 2:
            return nrows + ncols - 2

        """
        bfs from (0, 0)
        """

        queue = deque()
        seen = set()
        remaining = k

        queue.append((0, 0, 0, remaining))

        while queue:
            steps, x, y, remaining = queue.popleft()

            if x == nrows - 1 and y == ncols - 1:
                return steps

            if (x, y, remaining) in seen:
                continue

            seen.add((x, y, remaining))
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            valid = []
            for p in directions:
                nx = x + p[0]
                ny = y + p[1]
                if nx >= 0 and nx < nrows and ny >= 0 and ny < ncols:
                    valid.append((nx, ny))

            for neighbor in valid:
                nx = neighbor[0]
                ny = neighbor[1]

                if grid[nx][ny] == 0:
                    queue.append((steps + 1, nx, ny, remaining))
                else:
                    if remaining <= 0:
                        continue
                    queue.append((steps + 1, nx, ny, remaining - 1))

        return -1

