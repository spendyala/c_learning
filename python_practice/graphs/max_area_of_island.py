# https://leetcode.com/problems/max-area-of-island/

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:

        rows = len(grid)
        cols = len(grid[0])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        def get_neighbors(x, y):
            for each_x, each_y in directions:
                if 0<= x + each_x <= rows-1 and 0 <= y + each_y <= cols-1:
                    yield x + each_x, y + each_y

        max_ans = 0
        seen = set()

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    continue

                stack = [(i,j)]
                seen.add((i,j))
                ans = 1
                while stack:
                    temp_x, temp_y = stack.pop()
                    for neig_x, neig_y in get_neighbors(temp_x, temp_y):
                        if (neig_x, neig_y) not in seen and grid[neig_x][neig_y] == 1:
                            ans += 1
                            seen.add((neig_x, neig_y))
                            stack.append((neig_x, neig_y))
                max_ans = max(max_ans, ans)
        return max_ans