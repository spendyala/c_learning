//
// Created by Subbu Pendyala on 1/13/24.
//


#include <stdio.h>
#include <stdlib.h>


/**
 * `typedef` keyword is used to create an alias (a new name) for an existing data type.
 * `typedef struct node { ... } node_t;` is creating an alias `node_t` for `struct node`.
 * Instead of writing `struct node` we write `node_t`
 */
typedef struct node {
    int data;
    struct node* left;
    struct node* right;
} node_t;

node_t* create_new_node(int data);
void in_order(node_t* root);
void post_order(node_t* root);
void pre_order(node_t* root);
node_t* insert_bst(node_t* node, int key);
node_t* min_value_in_bst(node_t* node);
node_t* delete_node(node_t* root, int key);
node_t * search_bst(node_t* root, int key);

node_t* create_new_node(int data) {
    /**
     * Create a new node
     */
    node_t* node = (node_t*)malloc(sizeof(node_t));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}

void in_order(node_t* root) {
    /**
     * In-order traversal
     */
    if (root != NULL) {  // if (!root)
        in_order(root->left);
        printf("%d ", root->data);
        in_order(root->right);
    }
}

void pre_order(node_t* root) {
    /**
     * Pre-order traversal
     */
    if (root != NULL) {  // if (!root)
        printf("%d ", root->data);
        pre_order(root->left);
        pre_order(root->right);
    }
}

void post_order(node_t* root) {
    /**
     * Post-order traversal
     */
    if (root != NULL) {  // if (!root)
        post_order(root->left);
        post_order(root->right);
        printf("%d ", root->data);
    }
}

node_t* min_value_in_bst(node_t* node) {
    /**
     * Function to find the min value in a BST
     */
    node_t* current = node;
    while (current && current->left != NULL)
        current = current->left;
    return current;
}

node_t* insert_bst(node_t* node, int key) {
    /**
     * Insert node into a BST
     */
    // If the tree is empty, return a new node
    if(node == NULL) {
        return create_new_node(key);
    }

    // Recursively find the correct position for the new node
    if(key < node->data) {
        node->left = insert_bst(node->left, key);
    }
    else if (key > node->data){
        node->right = insert_bst(node->right, key);
    }

    return node;
}

node_t* delete_node(node_t* root, int key) {
    /**
     * Delete node in a BST
     *
     * 1. Node with no child (or Leaf node): Simply remove the node
     * 2. Node with one child: Replace the node with its child.
     * 3. Node with two children: Find the inorder successor (smallest in the right subtree)
     *      or inorder predecessor (largest in the left subtree),
     *      copy its value to the node, and delete the successor/predecessor
     */
    if (root == NULL) return root;

    // If the key is smaller than the root, then it is in left subtree
    if (key < root->data) {
        root->left = delete_node(root->left, key);
    }
        // If the key is greater than the root, then it is in right subtree
    else if (key > root->data) {
        root->right = delete_node(root->right, key);
    }
        // If key is same as root, then this is the node to be deleted
    else {
        // Node with only one child or no child
        if (root->left == NULL) {
            node_t* temp = root->right;
            free(root);
            return temp;
        }
        else if (root->right == NULL) {
            node_t* temp = root->left;
            free(root);
            return temp;
        }

        // Node with two children: Get the inorder successor (smallest in the right subtree)
        node_t* temp = min_value_in_bst(root->right);

        // Copy the inorder successor's content to this node
        root->data = temp->data;

        // Delete the inorder successor
        root->right = delete_node(root->right, temp->data);
    }
    return root;
}

node_t * search_bst(node_t* root, int key) {
    // Base Cases: root is null or key is present at root
    if (root == NULL || root->data == key)
        return root;

    // Key is greater than root
    if (root->data < key)
        return search_bst(root->right, key);

    // Key is smaller than root
    return search_bst(root->left, key);
}


int main() {
    node_t* root = NULL;
    root = create_new_node(2);
    insert_bst(root, 1);
    insert_bst(root, 3);
    insert_bst(root, 4);
    insert_bst(root, 5);
    insert_bst(root, 50);
    insert_bst(root, 30);
    insert_bst(root, 20);
    insert_bst(root, 25);
    insert_bst(root, 15);
    insert_bst(root, 10);
    insert_bst(root, 40);

    printf("In-order traversal: ");
    in_order(root);

    printf("\nPre-order traversal: ");
    pre_order(root);

    printf("\nPost-order traversal: ");
    post_order(root);

    printf("\nMin value in the BST: %d", min_value_in_bst(root)->data);


    root = delete_node(root, 20);
    printf("\nIn-order traversal after deleting 20: ");
    in_order(root);

    root = delete_node(root, 30);
    printf("\nIn-order traversal after deleting 30: ");
    in_order(root);

    root = delete_node(root, 50);
    printf("\nIn-order traversal after deleting 30: ");
    in_order(root);


    return 0;
}