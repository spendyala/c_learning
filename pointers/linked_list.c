//
// Created by Subbu Pendyala on 1/10/24.
//

#include <stdio.h>
#include <stdlib.h>

struct node {
    int data;
    struct node *next;
} node_global;

void print_linked_list(struct node *head);
void add_tail(struct node *head, int data);
void delete_linked_list(struct node *head);


int main(){
    struct node *head;
    struct node *tail = NULL;
    head = (struct node *) malloc(sizeof(struct node));
    head->data = 0;
    head->next = tail;
    print_linked_list(head);
    add_tail(head, 5);
    add_tail(head, 10);
    add_tail(head, 15);
    print_linked_list(head);
    delete_linked_list(head);
}

void delete_linked_list(struct node *head){
    if(head->next){
        delete_linked_list(head->next);
    }
    free(head);
}

void add_tail(struct node *head, int data){
    struct node *prev = head;
    struct node *new_node = (struct node *) malloc(sizeof(struct node));
    new_node->data = data;
    while(prev->next){
        prev = prev->next;
    }
    prev->next = new_node;
    new_node->next = NULL;
}

void print_linked_list(struct node *head){
    printf("%d -> ", head->data);
    while(head->next){
        head = head->next;
        printf("%d -> ", head->data);
    }
    printf("\n");
}