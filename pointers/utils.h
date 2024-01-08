//#ifndef UNTITLED_UTILS_H
//#define UNTITLED_UTILS_H
//
//#endif //UNTITLED_UTILS_H
//#include <stdio.h>
#include <stdlib.h>

int* one_d_array(int n){
    // Type /** + Enter
    int* array = (int*)malloc(n * sizeof(int));
    int i;
    for(i=0; i<n; i++){
        array[i] = i;
    }
    return array;
}

int** two_d_array(int row, int col){
    // Type /** + Enter
    int** array = (int**)malloc(row * sizeof(int*));
    int r, c;
    for(r=0; r<row; r++){
        array[r] = (int*)malloc(col * sizeof(int));
    }

    for(r=0; r<row; r++){
        array[r] = one_d_array(col);
    }
    return array;
}