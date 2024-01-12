#include <stdio.h>
#include <stdlib.h>
#include "utils.h"



int main() {
    /*
    int x = 1;
    int y = 2;
    int z[10] = {10, 20, 30, 40, 50};

    int td[2][2] = {{100, 200}, {300, 400}};
    */
    int i, n= 5;
    int* array = one_d_array(n);

    for(i=0; i<n; i++){
        printf("%d ", array[i]);
    }
    free(array);
    printf("\nEnd of one d array\n");


    int row = 3, col = 3;
    int r, c;
    int** array_2d = two_d_array(row, col);

    for(r=0; r < row; r++){
        for(c=0; c < col; c++){
            printf("%d ", array_2d[r][c]);
        }
        printf("\n");
    }
    for (r = 0; r < row; r++)
        free(array_2d[r]);
    free(array_2d);
    printf("\nEnd of two d array\n");

    /*
    int *ip;

    ip = &x;

    printf("%d\n", *ip);
    printf(ip);
    printf("\n");

    // Pointer to the Pointer, memory address
    printf("%s\n", (const char *) &ip);

    printf("%s\n", (const char *) *&ip);
    printf("%d\n", *&z[0]);
    printf("%d\n", *&z[2]);

    printf("%d\n", *td[0]);
    printf("%d\n", *td[1]);

    printf("%d\n", td[0]);
    printf("%d\n", td[1]);


    printf("%d\n", td+1);
    printf("%d\n", td[1]);
    */


    return 0;
}
