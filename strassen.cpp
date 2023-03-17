#include <iostream>
#include <vector>
#include <set>
#include <thread>
#include <time.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>

using namespace std;
int NUM_THREADS = 8;

// ./strassen 0 dimension inputfile
// Dimension = dimension of matrix
// The inputfile is an ASCII file with 2d^2 integer numbers, one per line, representing two matrices A and B; you are to find the product AB = C. The first integer number is matrix entry a0,0, followed by a0,1, a0,2, . . . , a0,d−1; next comes a1,0, a1,1, and so on, for the first d^2 numbers. The next d^2 numbers are similar for matrix B.

// Define a struct for matrix with the dimension and matrix
struct matrix {
    vector<vector<int>> mat;
};

// Matrix is [i][j] where i is row and j is column, starting at 0

// Functions to add/subtract two matrices given a third matrix to store the result
/**
void add(matrix* A, matrix* B, matrix* result) {
    int n = A->dim;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->mat[i][j] = A->mat[i][j] + B->mat[i][j];
        }
    }
}

void sub(matrix* A, matrix* B, matrix* result) {
    int n = A->dim;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->mat[i][j] = A->mat[i][j] - B->mat[i][j];
        }
    }
}
*/

// Function to add/subtract two sub-matrices of larger matrices given third matrix to store result
// This is more space-efficient and possibly faster than the above functions
// The _row and _col variables are the starting row and column of the sub-matrices that we want to add/subtract and populate
void add_efficient(matrix* A, matrix* B, matrix* result, int A_row, int A_col, int B_row, int B_col, int result_row, int result_col, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->mat[i + result_row][j + result_col] = A->mat[i + A_row][j + A_col] + B->mat[i + B_row][j + B_col];
        }
    }
}

void sub_efficient(matrix* A, matrix* B, matrix* result, int A_row, int A_col, int B_row, int B_col, int result_row, int result_col, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->mat[i + result_row][j + result_col] = A->mat[i + A_row][j + A_col] - B->mat[i + B_row][j + B_col];
        }
    }
}

// Functions to pad matrix with zeroes (2^k method and m * 2^k method where m is \leq the crossover point)

// Find m for the m * 2^k method
int find_m(int dimension, int crossover) {
    int i = 0;
    while (dimension > crossover) {
        if (dimension % 2) dimension++;
        dimension >>= 1;
        i++;
    }
    dimension <<= i;
    return dimension;
}

// find 2^k for the 2^k method
int find_2k(int dimension) {
    int log_dimension = log2(dimension);
    if (pow(2, log_dimension) == dimension) return (int)dimension;
    else return (int)(pow(2, log_dimension + 1));
}

// function to pad or unpad
void pad_unpad(matrix* A, int new_dimension) {
    A->mat.resize(new_dimension);
    for (int i = 0; i < new_dimension; i++) {
        A->mat[i].resize(new_dimension);
    }
}

// function to print diagonal entries of matrix
void print_diagonal(matrix* A, int limit) {
    for (int i = 0; i < limit; i++) {
        cout << A->mat[i][i] << endl;
    }
}

// conventional matrix multiplication
// swap k and j to increase cache efficiency as per https://www.hindawi.com/journals/misy/2022/9650652/
// 
void conventional(matrix* A, matrix* B, matrix* C, int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int size) {
    for (int i = 0; i < size; i++) 
        for (int k = 0; k < size; k++) 
            for (int j = 0; j < size; j++) 
                C->mat[i + C_row][j + C_col] += A->mat[i + A_row][k + A_col] * B->mat[k + B_row][j + B_col];
}

// conventional matrix multiplication with multithreading
void conventional_multithread_helper(matrix* A, matrix* B, matrix* C, int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int size, int size_per_thread, int thread_id) {
    int lower_limit = thread_id * size_per_thread;
    int upper_limit;

    if (lower_limit > size) return;

    if ((thread_id + 1) * size_per_thread < size)
        upper_limit = (thread_id + 1) * size_per_thread;
    else upper_limit = size;

    for (int i = lower_limit; i < upper_limit; i++) 
        for (int k = 0; k < size; k++) 
            for (int j = 0; j < size; j++) 
                C->mat[i + C_row][j + C_col] += A->mat[i + A_row][k + A_col] * B->mat[k + B_row][j + B_col];
}

void conventional_multithread(matrix* A, matrix* B, matrix* C, int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int size) {
    int size_per_thread = 0;
    if (size <= NUM_THREADS)
        size_per_thread = 1;
    else
        size_per_thread = 1 + ((size - 1) / NUM_THREADS);


    thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = thread(conventional_multithread_helper, A, B, C, A_row, A_col, B_row, B_col, C_row, C_col, size, size_per_thread, i);
    }
    // join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

// strassen singlethread - implementation from https://www.cise.ufl.edu/~sahni/papers/strassen.pdf
void strassen(matrix* A, matrix* B, matrix* C, int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int size) {
    // base case
    if (size == 1) {
        C->mat[C_row][C_col] = A->mat[A_row][A_col] * B->mat[B_row][B_col]; 
        return;
    }
    // recursive strassen: C is output, A and B are inputs
    // create temp matrices T1 and T2
    int submatrixsize = size / 2;
    matrix* T1 = new matrix();
    matrix* T2 = new matrix();
    pad_unpad(T1, submatrixsize);
    pad_unpad(T2, submatrixsize);

    // 1. C12 = A21 - A11
    sub_efficient(A, A, C, A_row + submatrixsize, A_col, A_row, A_col, C_row, C_col + submatrixsize, submatrixsize);

    // 2. C21 = B11 + B12
    add_efficient(B, B, C, B_row, B_col, B_row, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    // 3. C22 = C12 x C21 [recursive]
    strassen(C, C, C, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 4. C12 = A12 - A22
    sub_efficient(A, A, C, A_row, A_col + submatrixsize, A_row + submatrixsize, A_col + submatrixsize, C_row, C_col + submatrixsize, submatrixsize);

    // 5. C21 = B21 + B22
    add_efficient(B, B, C, B_row + submatrixsize, B_col, B_row + submatrixsize, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    // 6. C11 = C12 x C21 [recursive]
    strassen(C, C, C, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row, C_col, submatrixsize);

    // 7. C12 = A11 + A22
    add_efficient(A, A, C, A_row, A_col, A_row + submatrixsize, A_col + submatrixsize, C_row, C_col + submatrixsize, submatrixsize);

    // 8. C21 = B11 + B22
    add_efficient(B, B, C, B_row, B_col, B_row + submatrixsize, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    // 9. T1 = C12 x C21 [recursive]
    strassen(C, C, T1, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, 0, 0, submatrixsize);

    // 10. C11 = C11 + T1
    add_efficient(C, T1, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);

    // 11. C22 = C22 + T1
    add_efficient(C, T1, C, C_row + submatrixsize, C_col + submatrixsize, 0, 0, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 12. T2 = A21 + A22
    add_efficient(A, A, T2, A_row + submatrixsize, A_col, A_row + submatrixsize, A_col + submatrixsize, 0, 0, submatrixsize);

    // 13. C21 = T2 x B11 [recursive]
    strassen(T2, B, C, 0, 0, B_row, B_col, C_row + submatrixsize, C_col, submatrixsize);

    // 14. C22 = C22 - C21
    sub_efficient(C, C, C, C_row + submatrixsize, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 15. T1 = B21 - B11
    sub_efficient(B, B, T1, B_row + submatrixsize, B_col, B_row, B_col, 0, 0, submatrixsize);

    // 16. T2 = A22 x T1 [recursive]
    strassen(A, T1, T2, A_row + submatrixsize, A_col + submatrixsize, 0, 0, 0, 0, submatrixsize);

    // 17. C21 = C21 + T2
    add_efficient(C, T2, C, C_row + submatrixsize, C_col, 0, 0, C_row + submatrixsize, C_col, submatrixsize);

    // 18. C11 = C11 + T2
    add_efficient(C, T2, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);

    // 19. T1 = B12 - B22
    sub_efficient(B, B, T1, B_row, B_col + submatrixsize, B_row + submatrixsize, B_col + submatrixsize, 0, 0, submatrixsize);

    // 20. C12 = A11 x T1 [recursive]
    strassen(A, T1, C, A_row, A_col, 0, 0, C_row, C_col + submatrixsize, submatrixsize);

    // 21. C22 = C22 + C12
    add_efficient(C, C, C, C_row + submatrixsize, C_col + submatrixsize, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 22. T2 = A11 + A12
    add_efficient(A, A, T2, A_row, A_col, A_row, A_col + submatrixsize, 0, 0, submatrixsize);

    // 23. T1 = T2 x B22 [recursive]
    strassen(T2, B, T1, 0, 0, B_row + submatrixsize, B_col + submatrixsize, 0, 0, submatrixsize);

    // 24. C12 = C12 + T1
    add_efficient(C, T1, C, C_row, C_col + submatrixsize, 0, 0, C_row, C_col + submatrixsize, submatrixsize);

    // 25. C11 = C11 - T1
    sub_efficient(C, T1, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);

    // Deallocate memory
    delete(T1);
    delete(T2);
}

// strassen multithread
void strassen_multithread(matrix* A, matrix* B, matrix* C, int A_row, int A_col, int B_row, int B_col, int C_row, int C_col, int size) {
    // base case
    if (size == 1) {
        C->mat[C_row][C_col] = A->mat[A_row][A_col] * B->mat[B_row][B_col]; 
        return;
    }

    // recursive strassen: C is output, A and B are inputs
    // create temp matrices T1 and T2
    int submatrixsize = size / 2;
    matrix* T1 = new matrix();
    matrix* T2 = new matrix();
    pad_unpad(T1, submatrixsize);
    pad_unpad(T2, submatrixsize);

    // THREAD BLOCK 1

    // 1. C12 = A21 - A11
    {thread thread1(sub_efficient, A, A, C, A_row + submatrixsize, A_col, A_row, A_col, C_row, C_col + submatrixsize, submatrixsize);

    // 2. C21 = B11 + B12
    thread thread2(add_efficient, B, B, C, B_row, B_col, B_row, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    thread1.join();
    thread2.join();}

    // END THREADS

    // 3. C22 = C12 x C21 [recursive]
    strassen_multithread(C, C, C, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // THREAD BLOCK 2

    // 4. C12 = A12 - A22
    {thread thread3(sub_efficient, A, A, C, A_row, A_col + submatrixsize, A_row + submatrixsize, A_col + submatrixsize, C_row, C_col + submatrixsize, submatrixsize);

    // 5. C21 = B21 + B22
    thread thread4(add_efficient, B, B, C, B_row + submatrixsize, B_col, B_row + submatrixsize, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    thread3.join();
    thread4.join();}

    // END THREADS

    // 6. C11 = C12 x C21 [recursive]
    strassen_multithread(C, C, C, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row, C_col, submatrixsize);

    // THREAD BLOCK 3

    // 7. C12 = A11 + A22
    {thread thread5(add_efficient, A, A, C, A_row, A_col, A_row + submatrixsize, A_col + submatrixsize, C_row, C_col + submatrixsize, submatrixsize);

    // 8. C21 = B11 + B22
    thread thread6(add_efficient, B, B, C, B_row, B_col, B_row + submatrixsize, B_col + submatrixsize, C_row + submatrixsize, C_col, submatrixsize);

    thread5.join();
    thread6.join();}

    // END THREADS

    // 9. T1 = C12 x C21 [recursive]
    strassen_multithread(C, C, T1, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col, 0, 0, submatrixsize);

    // THREAD BLOCK 4

    // 10. C11 = C11 + T1
    {thread thread7(add_efficient, C, T1, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);

    // 11. C22 = C22 + T1
    thread thread8(add_efficient, C, T1, C, C_row + submatrixsize, C_col + submatrixsize, 0, 0, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 12. T2 = A21 + A22
    thread thread9(add_efficient, A, A, T2, A_row + submatrixsize, A_col, A_row + submatrixsize, A_col + submatrixsize, 0, 0, submatrixsize);

    thread7.join();
    thread8.join();
    thread9.join();}

    // END THREADS

    // 13. C21 = T2 x B11 [recursive]
    strassen_multithread(T2, B, C, 0, 0, B_row, B_col, C_row + submatrixsize, C_col, submatrixsize);

    // THREAD BLOCK 5

    // 14. C22 = C22 - C21
    {thread thread10(sub_efficient, C, C, C, C_row + submatrixsize, C_col + submatrixsize, C_row + submatrixsize, C_col, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 15. T1 = B21 - B11
    thread thread11(sub_efficient, B, B, T1, B_row + submatrixsize, B_col, B_row, B_col, 0, 0, submatrixsize);

    thread10.join();
    thread11.join();}
    
    // END THREADS

    // 16. T2 = A22 x T1 [recursive]
    strassen_multithread(A, T1, T2, A_row + submatrixsize, A_col + submatrixsize, 0, 0, 0, 0, submatrixsize);

    // THREAD BLOCK 6

    // 17. C21 = C21 + T2
    {thread thread12(add_efficient, C, T2, C, C_row + submatrixsize, C_col, 0, 0, C_row + submatrixsize, C_col, submatrixsize);

    // 18. C11 = C11 + T2
    thread thread13(add_efficient, C, T2, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);

    // 19. T1 = B12 - B22
    thread thread14(sub_efficient, B, B, T1, B_row, B_col + submatrixsize, B_row + submatrixsize, B_col + submatrixsize, 0, 0, submatrixsize);

    thread12.join();
    thread13.join();
    thread14.join();}

    // END THREADS

    // 20. C12 = A11 x T1 [recursive]
    strassen_multithread(A, T1, C, A_row, A_col, 0, 0, C_row, C_col + submatrixsize, submatrixsize);

    // THREAD BLOCK 7

    // 21. C22 = C22 + C12
    {thread thread15(add_efficient, C, C, C, C_row + submatrixsize, C_col + submatrixsize, C_row, C_col + submatrixsize, C_row + submatrixsize, C_col + submatrixsize, submatrixsize);

    // 22. T2 = A11 + A12
    thread thread16(add_efficient, A, A, T2, A_row, A_col, A_row, A_col + submatrixsize, 0, 0, submatrixsize);

    thread15.join();
    thread16.join();}

    // END THREADS

    // 23. T1 = T2 x B22 [recursive]
    strassen_multithread(T2, B, T1, 0, 0, B_row + submatrixsize, B_col + submatrixsize, 0, 0, submatrixsize);

    // THREAD BLOCK 8

    // 24. C12 = C12 + T1
    {thread thread17(add_efficient, C, T1, C, C_row, C_col + submatrixsize, 0, 0, C_row, C_col + submatrixsize, submatrixsize);

    // 25. C11 = C11 - T1
    thread thread18(sub_efficient, C, T1, C, C_row, C_col, 0, 0, C_row, C_col, submatrixsize);
    thread17.join();
    thread18.join();}

    // END THREADS

    // Deallocate memory
    delete(T1);
    delete(T2);
}

// generate random matrix
// type = 0 for {0, 1}, 1 for {0, 1, 2}, -1 for {0, 1, -1}
void generate_matrix(int type, int dimension, matrix* A) {
    // seed RNG with time
    srand((unsigned) time(NULL));

    for (int i = 0; i < dimension; i++) {
        A->mat.push_back(vector<int>());
        for (int j = 0; j < dimension; j++) {
            int random = rand();
            if (type == 0)
                A->mat[i].push_back(random % 2);
            else if (type == 1)
                A->mat[i].push_back(random % 3);
            else
                A->mat[i].push_back((random % 3) - 1);
        }
    }
}

int main(int argc, char** argv) {
    // ./strassen 0 dimension inputfile
    // If first argument is 0, A and B are taken from inputfile
    // If first argument is 1, A and B are randomly generated
    // If first argument is 2, A and B are written as a single string of single-digit integers

    // Start timer
    clock_t start, stop;
    start = clock();

    int matrixdimension = atoi(argv[2]);
    int totaldimension = matrixdimension * matrixdimension;
    int pad_limit = find_2k(matrixdimension);
    matrix A, B, C;

    if (atoi(argv[1]) == 2) { // Matrix input as string
        string matrices = argv[3];
        for (int i = 0; i < matrixdimension; i++) {
            A.mat.push_back(vector<int>());
            B.mat.push_back(vector<int>());
            for (int j = 0; j < matrixdimension; j++) {
                A.mat[i].push_back( (int)matrices[i * matrixdimension + j] - '0');
                B.mat[i].push_back( (int)matrices[i * matrixdimension + j + totaldimension] - '0');
            }
        }

    } else if (atoi(argv[1]) == 1) { // Generate random matrices

        int matrixtype = atoi(argv[3]);
        generate_matrix(matrixtype, matrixdimension, &A);
        generate_matrix(matrixtype, matrixdimension, &B);

    } else if (atoi(argv[1]) == 0) { // Interpret input file

        // "The inputfile is an ASCII file with 2d^2 integer numbers, one per line"
        ifstream inputfile(argv[3]);
        string line;

        int i = 0, j = 0;
        while (getline(inputfile, line)) {
            if (i < matrixdimension) {
                A.mat.push_back(vector<int>());
                B.mat.push_back(vector<int>());
            }

            if (i < totaldimension) {
                A.mat[j].push_back(stoi(line));
            } else {
                B.mat[j - matrixdimension].push_back(stoi(line));
            }
            i++;
            if (i != 0 && i % matrixdimension == 0)
                j++;
        }
        inputfile.close();
    }

    // Pad matrices to next power of 2
    pad_unpad(&A, pad_limit);
    pad_unpad(&B, pad_limit);
    pad_unpad(&C, pad_limit);

    // Matrix multiply using chosen method; blank = strassen normal, 1 = strassen multithread, 2 = conventional, 3 = conventional multithread, 4 = nothing (for tests)
    if (argv[4]) {
        int method = atoi(argv[4]);
        if (method == 1) {
            cout << "Strassen Multithread" << endl;
            strassen_multithread(&A, &B, &C, 0, 0, 0, 0, 0, 0, pad_limit);
        }
        else if (method == 2) {
            cout << "Conventional" << endl;
            conventional(&A, &B, &C, 0, 0, 0, 0, 0, 0, pad_limit);
        }
        else if (method == 3) {
            cout << "Conventional Multithread" << endl;
            conventional_multithread(&A, &B, &C, 0, 0, 0, 0, 0, 0, pad_limit);
        }
        else if (method == 4)
            return 0;
    }
    else
        strassen(&A, &B, &C, 0, 0, 0, 0, 0, 0, pad_limit);

    // Print C diagonals
    if (atoi(argv[1]) != 1)
        print_diagonal(&C, matrixdimension);        

    // free memory
    A.mat.clear();
    B.mat.clear();
    C.mat.clear();

    // stop timer
    stop = clock();
  
    // calculate time taken if random matrices used
    if (atoi(argv[1]) == 1) {
        double totaltime = double(stop - start) / double(CLOCKS_PER_SEC);
        cout << "Time taken (seconds): " << fixed << totaltime << setprecision(10) << endl;
    }

    return 0;
}