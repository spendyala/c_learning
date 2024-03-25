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

    def shortestPath_veera(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])

        if k >= m + n - 2: # Remove source and destination
            return m + n - 2

        def get_neighbors(x, y):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for each_x, each_y in directions:
                if 0<= x + each_x <= m-1 and 0 <= y + each_y <= n-1:
                    yield x + each_x, y + each_y

        seen = set()
        queue = deque()

        destination = (m-1, n-1)

        queue.append((0, 0, k, 0))

        while queue:
            i, j, eliminations_counter, distance = queue.popleft()

            if (i, j) == destination:
                return distance

            if (i, j, eliminations_counter) in seen:
                continue

            seen.add((i, j, eliminations_counter))

            for row, col in get_neighbors(i, j):
                if grid[row][col] == 0:
                    queue.append((row, col, eliminations_counter, distance+1))
                elif grid[row][col] == 1 and eliminations_counter > 0:
                    queue.append((row, col, eliminations_counter-1, distance+1))

        return -1

