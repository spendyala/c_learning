import bisect
class TreeNode():
    def __init__(self, data, left, right):
        self.data = data # unordered list
        self.left = left
        self.right = right

    def binary_search_bisect(self, array, element):
        # Ensure the list is sorted
        # array.sort() # commenting as we know our array is sorted.
        # Find the position where x should be inserted to maintain sorted order
        i = bisect.bisect_left(array, element)
        # Check if x exists at the index i
        if i != len(array) and array[i] == element:
            return 1  # Element exists
        return 0  # Element does not exist

    def search(self,  array, element):
        return 1 if element in array else 0

class BinarySearchVariant():
    def search(self, root: TreeNode, element: int, ordered_flag: bool):

        stack = [root]
        ans = 0

        while stack:
            node = stack.pop()
            if ordered_flag:
                ans += node.binary_search_bisect(element)
            else:
                ans += node.search(element)
            if root.left:
                stack.append(root.left)
            if root.right:
                stack.append(root.right)

        return ans



