# https://leetcode.com/problems/01-matrix/

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
