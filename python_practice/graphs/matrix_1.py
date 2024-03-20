# https://leetcode.com/problems/01-matrix/
from typing import List
from collections import deque

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        rows = len(mat)
        cols = len(mat[0])

        output = [[0]*cols for _ in range(rows)]

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        def get_neighbors(x, y):
            for each_x, each_y in directions:
                if 0<= x + each_x <= rows-1 and 0 <= y + each_y <= cols-1:
                    yield x + each_x, y + each_y

        seen = set()
        queue = []
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    seen.add((i, j))
                    queue.append((i, j))

        distance = 1
        while queue:
            length_queue = len(queue)
            for _ in range(length_queue):
                x, y = queue.pop(0)
                for each_x, each_y in get_neighbors(x, y):
                    if (each_x, each_y) not in seen:
                        output[each_x][each_y] = distance
                        seen.add((each_x, each_y))
                        queue.append((each_x, each_y))
            distance += 1
        return output

    def answer(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        queue = deque()

        # Do it in place.
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i, j))
                else:
                    mat[i][j] = float('inf')

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Update distance if distance + 1 is less than previous
                if 0 <= nx < m and 0 <= ny < n and mat[nx][ny] > mat[x][y] + 1:
                    mat[nx][ny] = mat[x][y] + 1
                    queue.append((nx, ny))

        return mat
