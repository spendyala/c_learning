//
// Created by Subbu Pendyala on 1/10/24.
//

#include <stdio.h>
#include <stdlib.h>

/**
 * defines a new structure type named `node`.
 * This `node` has two members
 * data: This is an integer variable that store some data value in each node.
 * next: This is a pointer to another structure of the same type (struct node).
 */
struct node {
    int data;
    struct node *next;
};

void print_linked_list(struct node *head);
void add_tail(struct node *head, int data);
void delete_linked_list(struct node **head);
void add_head(struct node **head, int data);
void add_at_position(struct node **head, int data, int position);
void delete_by_value(struct node **head, int data);
int search(struct node *head, int data);


int main(){
    // Linked List head
    struct node *head = NULL;

    // Linked List tail
    struct node *tail = NULL;
    head = (struct node *) malloc(sizeof(struct node));
    // head -> tail (link is created below)
    head->data = 0;
    head->next = tail;

    print_linked_list(head);
    add_tail(head, 1);
    add_tail(head, 2);
    print_linked_list(head);
    add_head(&head, -1);
    add_head(&head, -2);
    print_linked_list(head);
    add_at_position(&head, -3, 0);
    add_at_position(&head, 3, 6);
    print_linked_list(head);
    delete_by_value(&head, 0);
    print_linked_list(head);
    delete_linked_list(&head);
    printf("Position of %d is %d\n", 0, search(head, 0));
    printf("Position of %d is %d\n", -1, search(head, -1));
    printf("Position of %d is %d\n", 3, search(head, 3));
}

void delete_linked_list(struct node **head){
    struct node *current = *head;
    struct node *next;

    while (current != NULL) {
        next = current->next;
        free(current);
        current = next;
    }

    *head = NULL;
}

void add_head(struct node **head, int data) {
    struct node *new_node = (struct node *)malloc(sizeof(struct node));
    if (new_node == NULL) {
        // Handle memory allocation failure
        return;
    }
    new_node->data = data;
    new_node->next = *head;
    *head = new_node;
}


void add_tail(struct node *head, int data){
    /** prev is essential to traverse to the end, without altering the head */
    struct node *prev = head;
    struct node *new_node = (struct node *) malloc(sizeof(struct node));
    new_node->data = data;
    while(prev->next){
        prev = prev->next;
    }
    prev->next = new_node;
    new_node->next = NULL;
}

void add_at_position(struct node **head, int data, int position) {
    if (position < 0) {
        printf("Invalid position\n");
        return;
    }

    if (position == 0) {
        add_head(head, data);
        return;
    }

    struct node *new_node = (struct node *)malloc(sizeof(struct node));
    new_node->data = data;

    struct node *current = *head;
    for (int i = 0; i < position - 1 && current != NULL; i++) {
        current = current->next;
    }

    if (current == NULL) {
        printf("Position out of bounds\n");
        free(new_node);
    } else {
        new_node->next = current->next;
        current->next = new_node;
    }
}

void delete_by_value(struct node **head, int data) {
    if (*head == NULL) {
        return;
    }

    struct node *temp = *head, *prev = NULL;
    if (temp != NULL && temp->data == data) {
        *head = temp->next;
        free(temp);
        return;
    }

    while (temp != NULL && temp->data != data) {
        prev = temp;
        temp = temp->next;
    }

    if (temp == NULL) return;

    prev->next = temp->next;
    free(temp);
}

int search(struct node *head, int data) {
    struct node *current = head;
    int position = 0;
    while (current != NULL) {
        if (current->data == data) {
            return position;
        }
        current = current->next;
        position++;
    }
    return -1; // Data not found
}

void print_linked_list(struct node *head){
    if (head == NULL) {
        printf("List is empty\n");
        return;
    }

    printf("HEAD -> ");
    struct node *current = head;
    while (current) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("TAIL\n\n");
}

