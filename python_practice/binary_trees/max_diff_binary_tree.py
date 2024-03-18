# https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def find_max_min(self, root):

        min_val = 0
        max_val = 0
        stack = [(root, min_val, max_val)]

        while stack:
            node, temp_min, temp_max = stack.pop()
            min_val = min(min_val, node.val)
            max_val = max(max_val, node.val)
            if node and node.right is not None:
                stack.append((node.right, min_val, max_val))
            if node and node.left is not None:
                stack.append((node.left, min_val, max_val))
        return min_val, max_val

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        '''
        '''

        if root is None:
            return 0
        abs_max_right = 0
        abs_min_right = 0
        local_right = 0
        local_left = 0

        if root and root.right is not None:
            min_right, max_right = self.find_max_min(root.right)
            abs_max_right = max(abs(max_right - root.val), abs(min_right - root.val))
            local_right = self.maxAncestorDiff(root.right)

        if root and root.left is not None:
            min_left, max_left = self.find_max_min(root.left)
            abs_max_left = max(abs(max_left - root.val), abs(min_left - root.val))
            local_left = self.maxAncestorDiff(root.left)

        return max(abs_max_right, abs_max_left, local_right, local_left)