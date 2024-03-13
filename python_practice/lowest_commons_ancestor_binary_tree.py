# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
            - node p and not other node is not visited then node is the ancestor
            - we are at the node, one of the leaf is p or q and other one has been visited.

            - node is p or q we return Node

        '''
        if not root:
            return None

        stack = [root]
        visited = {p: False, q: False}
        if root.val == p.val:
            visited[p] = True
            return root
        if root.val == q.val:
            visited[q] = True
            return root


        while stack:
            node = stack.pop()

            if node.val == p.val and not visited[q]:
                visited[p] = True
                return node
            if node.val == q.val and not visited[p]:
                visited[q] = True
                return node

            if node.left and node.left.val in {p.val, q.val}:
                if not all((visited[p], visited[q])):
                    return node.left

            if node.right and node.right.val in {p.val, q.val}:
                if not all((visited[p], visited[q])):
                    return node.right
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return None