#include <Eigen/Core>
#include <iostream>

#include <typeinfo>

#define MATRIX_SIZE 10

/**********************************************
Given a large Eigen matrix, 
extract the values in the top 3x3 block,
and then assign the top 3x3 block to be an identity matrix.
**********************************************/

// Useful link: https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html


// Method 1: using two layers of for loop to extract the values and change the values
void upperleft_replace_method1(Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> * big_matrix, Eigen::Matrix3d * topleft_block) {
    for (int i = 0; i < 3; i++) { // i for row
        for (int j = 0; j < 3; j++){ // j for column
            (*topleft_block)(i,j) = (*big_matrix)(i,j);
            if (i == j) {
                (*big_matrix)(i,j) = 1; // parentheses to dereference first, then access by index
            }
            else {
                (*big_matrix)(i,j) = 0;
            }
        }
    }
}


// Method 2: using the .block method for block operation to exact and modify values
void upperleft_replace_method2(Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> * big_matrix, Eigen::Matrix3d * topleft_block) {
    *topleft_block = (*big_matrix).block(0,0,3,3);
    (*big_matrix).block(0,0,3,3) = Eigen::Matrix3d::Identity();
}


int main(int argc, char **argv) {
    // only showing upto four decimal space
    std::cout.precision(4);
    
    // Define a random matrix with dimension of [MATRIX_SIZE x MATRIX_SIZE]
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);

    std::cout << "The initial matrix is: \n" << matrix_NN << std::endl;

    // Declare an empty 3x3 matrix to restore the top-left block of the big matrix
    Eigen::Matrix3d topleft_block;

    // Call the matrix replacement function to extract the upper-left 3x3 matrix block
    
    // Method 1: using two layers of for loop to extract the values and change the values
    // upperleft_replace_method1(&matrix_NN, &topleft_block);

    // Method 2: using the .block method for block operation to exact and modify values
    upperleft_replace_method2(&matrix_NN, &topleft_block);

    // print out the resultant matrices
    std::cout << "\n The upper left block is: \n" << topleft_block << std::endl;
    std::cout << "\n The modified matrix is: \n" << matrix_NN << std::endl;

    return 0;
}

