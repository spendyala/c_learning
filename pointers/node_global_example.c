//
// Created by Subbu Pendyala on 1/13/24.
//

#include <stdio.h>
#include <stdlib.h>

/**
 * defines a new structure type named `node`.
 * This `node` has two members
 * data: This is an integer variable that store some data value in each node.
 * next: This is a pointer to another structure of the same type (struct node).
 *
 * node_global: A global variable named node_global of the type (struct node).
 * node_global is accessible from any part of the program.
 * node_global variable will have the data and next members.
 * Global variable, is statically allocated.
 *
 * Global variable lifetime extends across the entire runtime of the program.
 * It is automatically allocated and deallocated only by the system.
 *
 * We can't use free() to delete the statically allocated memory.
 *
 * free() is used on dynamically allocated memory by malloc, calloc, realloc
 */
struct node {
    int data;
    struct node *next;
} node_global;

int main() {
    // Initialize node_global
    node_global.data = 1;
    node_global.next = NULL;

    // Create new_node dynamically
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = 2;
    new_node->next = NULL;

    // Linking node_global to new_node
    node_global.next = new_node;

    printf("node_global data: %d\n", node_global.data);
    printf("new_node data: %d\n", new_node->data);

    // Free dynamically allocated memory
    free(new_node);

    return 0;
}
