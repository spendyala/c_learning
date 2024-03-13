# https://leetcode.com/problems/range-sum-of-bst/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rangeSumBST_recursive(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0
        result = 0
        if low <= root.val <= high:
            result += root.val
        if root.val < high:
            result += self.rangeSumBST_recursive(root.right, low, high)
        if root.val > low:
            result += self.rangeSumBST_recursive(root.left, low, high)

        return result

    def rangeSumBST_success(self, root: Optional[TreeNode], low: int, high: int) -> int:
        # if root is None
        if not root:
            return 0
        result = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if low <= node.val <= high:
                result += node.val
            if node.right and node.val < high:
                stack.append(node.right)
            if node.left and node.val > low:
                stack.append(node.left)

        return result

    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        # if root is None
        # if not root:
        #     return 0
        result = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if low <= node.val <= high:
                result += node.val
            if node.right and node.val < high:
                stack.append(node.right)
            if node.left and node.val > low:
                stack.append(node.left)

        return result
