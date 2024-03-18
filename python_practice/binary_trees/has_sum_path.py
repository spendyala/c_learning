# https://leetcode.com/problems/path-sum/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def hasPathSum_1(self, root: TreeNode, sum_: int) -> bool:
        if not root:
            return False
        sum_ -= root.val
        if not root.left and not root.right:
            return sum_ == 0
        return self.hasPathSum(root.left,sum_) or self.hasPathSum(root.right,sum_)

    def hasPathSum_2(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, curr):
            if not node:
                return False

            # if both children are null, then the node is a leaf
            if node.left == None and node.right == None:
                return (curr + node.val) == targetSum

            curr += node.val
            left = dfs(node.left, curr)
            right = dfs(node.right, curr)
            return left or right

        return dfs(root, 0)

    def hasPathSum_3(self, root: Optional[TreeNode], targetSum: int) -> bool:
            if not root:
                return False

            stack = [(root, 0)]
            while stack:
                node, curr = stack.pop()
                # if both children are null, then the node is a leaf
                if node.left == None and node.right == None:
                    if (curr + node.val) == targetSum:
                        return True

                curr += node.val
                if node.left:
                    stack.append((node.left, curr))
                if node.right:
                    stack.append((node.right, curr))

            return False