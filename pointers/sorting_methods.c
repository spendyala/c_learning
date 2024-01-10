//
// Created by Subbu Pendyala on 1/10/24.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 1000
#define one 1
#define INFINITY 99999

void printseq(int *a,int n);
void insertion_sort(int a[],int n);
void selection_sort(int a[],int n);
void swap(int a[],int i, int j);

void merge_fun(int a[],int n);
void merge_sort(int a[],int n);
void merge(int a[],int left[],int right[],int li,int ri);

void quick_fun(int a[],int n);
void quick_sort(int a[],int n);

main(int argc, char* argv[]){
    int a[1000];
    int i;
    int n;

    printf("Number of integers to scan: ");
    scanf("%d",&n);
    for(i=0;i<n;i++){
        printf("Enter number %d: ",i+one);
        scanf("%d",&a[i]);
    }
    //insertion_sort(a,n);
    //selection_sort(a,n);
    //merge_fun(a,n);
    quick_fun(a,n);
}
/* ** Insertion Sort **
  Data Structure          : Array
  Worst Case Performance  : O(n^2)
  Best Case Performance   : O(n)
  Average Case Performance: O(n^2)
  Worst Case space Complexity: O(n) total, O(1) auxiliary
*/
void insertion_sort(int a[],int n){
    int i,j,key;
    printf("Before insertion sort:\n");
    printseq(&a[0],n);
    for(j=1;j<n;j++){
        key=a[j];
        i=j-1;
        for(;((i>=0)&&(a[i]>key));i--)
            a[i+1]=a[i];
        a[i+1]=key;
    }
    printf("Insertion sort result: \n");
    printseq(&a[0],n);
}
/* ** Selection Sort **
  Data Structure          : Array
  Worst Case Performance  : O(n^2)
  Best Case Performance   : O(n^2)
  Average Case Performance: O(n^2)
  Worst Case space Complexity: O(n) total, O(1) auxiliary
*/
void selection_sort(int a[],int n){
    int i,j,key,iMin;
    printf("Before selection sort:\n");
    printseq(&a[0],n);
    for(i=0;i<n;i++){
        iMin=i;
        for(j=i+1;j<n;j++){
            if(a[j]<a[iMin])
                iMin=j;
        }
        if(iMin!=i)
            swap(&a[0],iMin,i);
    }
    printf("Selection Sort result: \n");
    printseq(&a[0],n);
}
/* ** Merge sort **
  Data Structure          : Array
  Worst Case Performance  : O(n log n)
  Best Case Performance   : O(n log n) or O(n) natural variant
  Average Case Performance: O(n log n)
  Worst Case space Complexity: O(n) auxiliary
*/
void merge_fun(int a[],int n){
    int i;
    printf("Before Merge Sort: \n");
    printseq(&a[0],n);
    merge_sort(a,n);
    printf("Merge Sort result: \n");
    printseq(&a[0],n);
}
void merge_sort(int a[],int n){
    int middle,i,k;
    int left[n/2];
    int right[n-n/2];
    middle=n/2;
    if(middle<1)
        return;
    for(i=0;i<middle;i++){
        left[i]=a[i];
    }
    for(i=0;i<(n-middle);i++){
        right[i]=a[middle+i];
    }
    merge_sort(left,middle);
    merge_sort(right,n-middle);
    merge(a,left,right,middle,n-middle);
}
void merge(int a[],int left[],int right[],int li,int ri){
    int result[li+ri];
    int i=0,j=0,k=0;
    while((i<li)&&(j<ri)){
        if(left[i]<=right[j]){
            a[k++]=left[i++];
        }else{
            a[k++]=right[j++];
        }
    }
    while((i>=li)&&(j<ri)){
        a[k++]=right[j++];
    }
    while((j>=ri)&&(i<li)){
        a[k++]=left[i++];
    }
}
/* ** Quick sort **
  Data Structure             : Array
  Worst Case Performance     : O(n^2)
  Best Case Performance      : O(n log n) or O(n) natural variant
  Average Case Performance   : O(n log n)
  Worst Case space Complexity: O(n) auxiliary
*/
void quick_fun(int a[],int n){
    printf("Before Quick Sort: \n");
    printseq(&a[0],n);
    quick_sort(a,n);
    printf("Quick Sort result: \n");
    printseq(&a[0],n);
}
void quick_sort(int a[],int n){
    int i,pointer;
    for(i=0;i<n;i++)
        if(a[i]<a[0])
            swap(a,++pointer,i);
    swap(a,0,pointer);
    quick_sort(a,pointer);
    quick_sort(a+pointer+1,n-pointer-1);
}
//Swap
void swap(int *a,int i, int j){
    a[j]=a[i]+a[j];
    a[i]=a[j]-a[i];
    a[j]=a[j]-a[i];
}
//Printing sequence
void printseq(int *a,int n){
    int i;
    printf("Printing sequence:\n");
    for(i=0;i<n;i++)
        printf("%d ",a[i]);
    printf("\n");
}
