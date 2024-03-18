# https://leetcode.com/problems/minimum-depth-of-binary-tree/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def minDepth_recurssive(self, root: Optional[TreeNode]) -> int:

        if root is None: # root.left is None
            return 0

        # if root.left is None and root.right is None:
        #     return 1

        if root.left is None:
            return self.minDepth_recurssive(root.right) + 1
        if root.right is None:
            return self.minDepth_recurssive(root.left) + 1

        return min(self.minDepth_recurssive(root.left), self.minDepth_recurssive(root.right))+1

    def minDepth(self, root: Optional[TreeNode]) -> int:

        if root is None:
            return 0

        stack = [(root, 1)]
        min_val = (10**5) + 1

        # if root.left is None and root.right is None:
        #     return 1

        while stack:
            node, val = stack.pop()

            if node and node.left is None and node.right is None:
                min_val = min(min_val, val)

            if node and node.left is not None:
                stack.append((node.left, val+1))
            if node and root.right is not None:
                stack.append((node.right, val+1))

        return min_val
