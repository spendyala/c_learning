# https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        if not root:
            return 0

        def add_parent_recursive(node):
            if node:
                if node.right:
                    node.right.parent = node
                    add_parent_recursive(node.right)
                if node.left:
                    node.left.parent = node
                    add_parent_recursive(node.left)
            # if node == target:
            #     return
        add_parent_recursive(root)


        seen = set()
        queue = []

        seen.add(target)
        queue.append(target)
        level = 0

        while queue:
            if level == k:
                break
            level_length = len(queue)
            for _ in range(level_length):
                node = queue.pop(0)
                if not hasattr(node, 'parent'):
                    node.parent = None
                for each_node in [node.left, node.right, node.parent]:
                    if each_node and each_node not in seen:
                        seen.add(each_node)
                        queue.append(each_node)
            level += 1
        return [each_node.val for each_node in queue]