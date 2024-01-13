//#ifndef UNTITLED_UTILS_H
//#define UNTITLED_UTILS_H
//
//#endif //UNTITLED_UTILS_H
//#include <stdio.h>
#include <stdlib.h>

int* one_d_array(int n){
    // Type /** + Enter
    // https://arthursonzogni.com/Diagon/#Flowchart To generate the flow chart
    /**
          ┌────────────────────────┐
          │The purpose is to create│
          │a one-dimensional array │
          │of size `n`             │
          └────────────┬───────────┘
         ┌─────────────▽─────────────┐
         │This function takes an     │
         │integer `n` and returns a  │
         │pointer to an integer array│
         └─────────────┬─────────────┘
             ┌─────────▽────────┐
             │int*              │
             │one_d_array(int n)│
             └─────────┬────────┘
         ┌─────────────▽─────────────┐
         │int* array = (int*)malloc(n│
         │* sizeof(int));            │
         └─────────────┬─────────────┘
┌──────────────────────▽─────────────────────┐
│Dynamically allocates memory for n integers.│
│The malloc function allocates n *           │
│sizeof(int) bytes and returns a pointer to  │
│the allocated memory. (int*) type casts the │
│allocated memory as int type for each block.│
└──────────────────────┬─────────────────────┘
      ┌────────────────▽────────────────┐
      │The for loop (for(i=0; i<n; i++))│
      │initializes each element of the  │
      │array array[i] with its index I. │
      └────────────────┬────────────────┘
        ┌──────────────▽─────────────┐
        │Returns the pointer to the  │
        │first element of the        │
        │dynamically allocated array.│
        └────────────────────────────┘
     */

    int* array = (int*)malloc(n * sizeof(int));
    int i;
    for(i=0; i<n; i++){
        array[i] = i;
    }
    return array;
}

int** two_d_array(int row, int col){
    // Type /** + Enter
    /**
                 ┌─────────────────────────┐
                 │Function: two_d_array(int│
                 │row, int col)   (e.g.,   │
                 │row = 3, col = 4)        │
                 └────────────┬────────────┘
                ┌─────────────▽─────────────┐
                │int** array = malloc(3 *   │
                │sizeof(int*)); Memory      │
                │Allocation for Row Pointers│
                └─────────────┬─────────────┘
     ┌────────────────────────▽────────────────────────┐
     │                                                 │
     │ [ ][ ][ ]                                       │
     │  ^  ^  ^                                        │
     │  |  |  |                                        │
     │  |  |  +-- array[2] (points to an integer array)│
     │  |  +----- array[1] (points to an integer array)│
     │  +-------- array[0] (points to an integer array)│
     └──────────────────────┬──────────────────────────┘
     ┌──────────────────────▽──────────────────────┐
     │ Initialize Each Row Using one_d_array       │
     │  array[0] = one_d_array(4)                  │
     │  array[1] = one_d_array(4)                  │
     │  array[2] = one_d_array(4)                  │
     └──────────────────────┬──────────────────────┘
   ┌────────────────────────▽────────────────────────┐
   │                                                 │
   │  [0][1][2][3]   [0][1][2][3]   [0][1][2][3]     │
   │   ^  ^  ^  ^     ^  ^  ^  ^     ^  ^  ^  ^      │
   │   |  |  |  |     |  |  |  |     |  |  |  |      │
   │  array[0]       array[1]       array[2]         │
   └────────────────────────┬────────────────────────┘
              ┌─────────────▽────────────┐
              │ //Return Pointer to Array│
              │of Pointers               │
              │ return array;            │
              └──────────────────────────┘
     */
    int** array = (int**)malloc(row * sizeof(int*));
    int r;


// The below code will lead to memory leak and the memory is never freed.
//    for(r=0; r<row; r++){
//        array[r] = (int*)malloc(col * sizeof(int));
//    }

    for(r=0; r<row; r++){
        array[r] = one_d_array(col);
    }
    return array;
}