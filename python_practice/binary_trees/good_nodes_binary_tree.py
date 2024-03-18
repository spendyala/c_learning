# https://leetcode.com/problems/count-good-nodes-in-binary-tree/

class Solution:
    def goodNodes_1(self, root: TreeNode) -> int:
        def dfs(node, max_so_far):
            if not node:
                return 0

            left = dfs(node.left, max(max_so_far, node.val))
            right = dfs(node.right, max(max_so_far, node.val))
            ans = left + right
            if node.val >= max_so_far:
                ans += 1

            return ans

        return dfs(root, float("-inf"))

    def goodNodes_2(self, root: TreeNode) -> int:
        if not root:
            return 0

        stack = [(root, float("-inf"))]
        ans = 0

        while stack:
            node, max_so_far = stack.pop()
            if node.val >= max_so_far:
                ans += 1

            if node.left:
                stack.append((node.left, max(max_so_far, node.val)))
            if node.right:
                stack.append((node.right, max(max_so_far, node.val)))

        return ans