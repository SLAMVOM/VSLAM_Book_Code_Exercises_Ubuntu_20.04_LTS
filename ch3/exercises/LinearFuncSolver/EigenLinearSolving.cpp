/***********************************************************************
 * Questions:
 * When does a general linear equation Ax=b has a unique solution of x
 * Implement the different method in Eigen to solve for a linear system
 
 * According to:
 * https://www.math.arizona.edu/~lega/322/Spring07/Linear_Algebra_5_8_1_Handout_1x2.pdf
 * 
 * The system Ax = b has a unique solution provided dim(N(A)) = 0
 * (i.e., the dimension of null space of A is 0)
 * If the number of columns of A is n, then the system Ax = b has a 
 * unique solution if an only if rank(A) = n.
 *
 * If b is not in the column space of A, then the system Ax=b has no
 * solution.
 *
 * If the null space of A is non-trivial, then the system Ax=b has
 * more than one solution.
***********************************************************************/

/*
There are two categories of methods to compute a solution for a general linear system:
1. direct methods
2. iterative methods
*/

// Code reference: https://www.cnblogs.com/newneul/p/8306442.html

#include <iostream>
#include <ctime>
#include <cmath>
#include <complex>
#include <typeinfo>

// Eigen libraries
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

// Several methods require the A matrix to be a square matrix, only 1.2, 1.6, and 1.7 do not require a square matrix A
#define LINEAR_EQUATION_NUM 6 // the num of equations (or conditions) in the system, i.e., num of rows of A
#define VARIABLE_NUM 6 // the number of variables in the system, i.e., num of columns of A

int USE_SYMMETRIC_MAT = 1; // whether using a symmetric matrix when a random square matrix is used, true if > 0

typedef Eigen::Matrix<double, LINEAR_EQUATION_NUM, VARIABLE_NUM> MatA; // the coefficients matrix
typedef Eigen::Matrix<double, LINEAR_EQUATION_NUM, 1> VecB; // the RHS of the linear equation system
typedef Eigen::Matrix<double, VARIABLE_NUM, 1> VecX; // the unknown variable matrix

// The iterative process for the iterative solutions from Gauss-Seidel and Jacobi method
void iterativeProcess(const MatA &A_mat, const VecB &b_vec, VecX &x_vec, 
                      const int num_iterations, const double accuracy_level, 
                      const int method_code);

// Define the user-specified thresholds for the iterative methods
int num_iterations = 200;
double accuracy_level = 1e-9;


int main(int argc, char **argv) {
    // set the precision size of the output
    std::cout.precision(9);
    // set the variables
    MatA A_mat = Eigen::MatrixXd::Random(LINEAR_EQUATION_NUM, VARIABLE_NUM); // random number uniformly drawn from [-1,1]
    VecB b_vec = Eigen::MatrixXd::Random(LINEAR_EQUATION_NUM, 1);
    if (LINEAR_EQUATION_NUM == VARIABLE_NUM) {
        A_mat += Eigen::Matrix<double, LINEAR_EQUATION_NUM, VARIABLE_NUM>::Identity() * 3.0; // make sure the diagonal is large
        if (USE_SYMMETRIC_MAT > 0) {
            A_mat = A_mat.transpose() * A_mat; // inner product of the transpose and original matrix makes A_mat be a symmetric matrix
        }
    }

    // set the solution variable
    Eigen::Matrix<double, VARIABLE_NUM, 1> x_vec;

    // testing example
    
    // // case 1 (square matrices): solution: -2, 1, 1
    // A_mat << 1,2,3, 4,5,6, 7,8,10;
    // b_vec << 3, 3, 4;
    
    // // case 2 (square matrices): solution: 1, 1, 1
    // A_mat << 10,3,1, 2,-10,3, 1,3,10;
    // b_vec << 14, -5, 14;

    // case 3 (symmetric matrices):
    // A_mat << 4,2,1, 2,5,3, 1,3,7;
    // b_vec << 20, 40, 30;

    // set a time variable to count the time spent to find a solution
    clock_t tic = clock(); 

// References: 
// * https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
// * Chapra_2011_Applied numerical method with MATLAB for engineers and scientists
// * Yang_2020_Applied numerical methods using MATLAB

// 1.1 Direct inversion of matrix A to solve for x = A^-1 b
// This method will only be able to use when A is a square matrix,
// and if the matrix is not positive definite, there will not be a correct solution
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    x_vec = A_mat.inverse() * b_vec;
    std::cout << "The time spent by the direct inversion method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
#else
    std::cout << "The direct inversion method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif
    std::cout << "\n" << std::endl;


// 1.2 Pseudo-inverse of matrix A to solve for x = (A^T A)^-1 A^T b
// This method does NOT require A to be a square matrix. If A has full column rank (i.e., rank(A) = n, where A has n columns)
    tic = clock();
    x_vec = (A_mat.transpose() * A_mat).inverse() * A_mat.transpose() * b_vec;
    std::cout << "The time spent by the pseudo-inverse method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
    std::cout << "\n" << std::endl;


// 1.3 Gauss-Elimination (with or without pivoting), here implements a naive method. For GE with pivoting, see MATLAB example:
// Chapra_2011_Applied numerical method with MATLAB for engineers and scientists 
// This method requires the matrix A to be a square matrix.
// The process consists of two steps: 1) forward eliminator; 2) back substitution
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    // construct an augmented matrix
    Eigen::Matrix<double, LINEAR_EQUATION_NUM, VARIABLE_NUM + 1> Aug_mat;
    Aug_mat.block<LINEAR_EQUATION_NUM, VARIABLE_NUM>(0,0) = A_mat;
    Aug_mat.block<LINEAR_EQUATION_NUM, 1>(0,VARIABLE_NUM) = b_vec;

    // forward elimination
    for (int i = 0; i < VARIABLE_NUM-1; i++){
        for (int j = i+1; j < VARIABLE_NUM; j++) {
            double factor = Aug_mat(j,i) / Aug_mat(i,i);
            for (int k = i; k < VARIABLE_NUM+1; k++) {
                Aug_mat(j, k) = Aug_mat(j, k) - factor * Aug_mat(i, k);
            }
        }
    }
    // back substitution
    x_vec.setZero();
    x_vec[VARIABLE_NUM-1] = Aug_mat(VARIABLE_NUM-1,VARIABLE_NUM) / Aug_mat(VARIABLE_NUM-1,VARIABLE_NUM-1);
    for (int i = VARIABLE_NUM-2; i > -1; i--) {
        double tmp_val = ((Aug_mat.block(i, i+1, 1, VARIABLE_NUM-i-1)) * x_vec.tail(VARIABLE_NUM-i-1))[0];
        x_vec[i] = ( Aug_mat(i, VARIABLE_NUM) - tmp_val ) / (Aug_mat(i,i));
    }
    std::cout << "The time spent by the naive Gauss-Elimination method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
#else
    std::cout << "The Gauss Elimination method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif
    std::cout << "\n" << std::endl;


// 1.4 LU factorization method with partial pivoting by Eigen
// This method requires the matrix A to be an invertible matrix, thus square.
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    x_vec = A_mat.partialPivLu().solve(b_vec); // equivalently, A_mat.lu().solve(b_vec);
    std::cout << "The time spent by the partial pivot LU method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
#else
    std::cout << "The LU  partial pivoting method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif
    std::cout << "\n" << std::endl;


// 1.5 Robust Cholesky factorization method by Eigen
// This method requires the matrix A to be a positive or negative semidefinite matrix, thus must be a symmetric matrix
// Note: the result by this method can contain noticible error (at least for a small linear equation system)
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    x_vec = A_mat.ldlt().solve(b_vec); //
    std::cout << "The time spent by the partial pivot Robust Cholesky method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
    std::cout << "Note: This method requires the matrix A to be positive or negative semidefinite. If not, the result is not accurate." <<std::endl;
#else
    std::cout << "The Robust Cholesky method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif
    std::cout << "\n" << std::endl;


// 1.6 QR factorization method by Eigen
// This method does not require the matrix to be square.
// As recommended by Eigen, if the A matrix is full rank, the HouseHolderQR (without pivoting) is the method of choice.
    tic = clock();
    x_vec = A_mat.householderQr().solve(b_vec); //
    std::cout << "The time spent by the HouseHolderQR method (fastest): " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;

    tic = clock();
    x_vec = A_mat.colPivHouseholderQr().solve(b_vec); //
    std::cout << "The time spent by the HouseHolderQR with column-pivoting method (intermediate): " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;

    tic = clock();
    x_vec = A_mat.fullPivHouseholderQr().solve(b_vec); //
    std::cout << "The time spent by the HouseHolderQR with full-pivoting method (slowest): " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
    std::cout << "\n" << std::endl;


// 1.7 Singular Value Decomposition (SVD) method by Eigen
// This method does not require the matrix to be square.
// Note: The thin U and V are only available when the matrix has a dynamic number of columns
    Eigen::MatrixXd A_svd = A_mat * 1.0;
    Eigen::MatrixXd b_svd = b_vec * 1.0;
    tic = clock();
    x_vec = A_svd.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_svd); //
    std::cout << "The time spent by the (Bidiagonal Divide and Conquer) BDCSVD method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;

    tic = clock();
    x_vec = A_svd.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_svd); //
    std::cout << "The time spent by the (Two-sided Jacobi) JacobiSVD method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
    std::cout << "\n" << std::endl;


// The follows are the iterative methods of Gauss-Seidel and Jacobi techniques.
// The convergent condition for both methods requires the spectral radius, rho, to be smaller than 1.
// A detailed description of the convergent conditions of the two methods may refer to:
// Burden_2011_Numerical Analysis_Ed9_Sec. 7.3

// 2.1 Jacobi Iteration
// This method requires the matrix A to be a square matrix.
// Implementation reference: https://www.cnblogs.com/newneul/p/8306442.html
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    iterativeProcess(A_mat, b_vec, x_vec,
                     num_iterations, accuracy_level,
                     1); // 1-Jacobi; 2-Gauss-Seidel
    std::cout << "The time spent by the Jacobi Iteration method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
#else
    std::cout << "The Jacobi Iteration method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif
    std::cout << "\n" << std::endl;


// 2.2 Jacobi Iteration
// This method requires the matrix A to be a square matrix.
// Implementation reference: https://www.cnblogs.com/newneul/p/8306442.html
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
    tic = clock();
    iterativeProcess(A_mat, b_vec, x_vec,
                     num_iterations, accuracy_level,
                     2); // 1-Jacobi; 2-Gauss-Seidel
    std::cout << "The time spent by the Gauss-Seidel method: " << 1000 * (clock()-tic)/(double)CLOCKS_PER_SEC
        << "ms" << std::endl << "Solution is:\t" << x_vec.transpose() << std::endl;
#else
    std::cout << "The Gauss-Seidel method cannot solve this system. (Note: the number of linear equations and variables must be the same in this method)" << std::endl;
#endif

} // end of main()


// if the iterative process does not converge, return zero vector as solution
// Note the input A_mat is a square matrix
// Only compile the following code when: LINEAR_EQUATION_NUM == VARIABLE_NUM
#if (LINEAR_EQUATION_NUM == VARIABLE_NUM)
void iterativeProcess(const MatA &A_mat, const VecB &b_vec, VecX &x_vec,
                      const int num_iterations, const double accuracy_level,
                      const int method_code     // 1 - Jacobi; 2 - Gauss-Seidel
                     )
{
    x_vec = Eigen::Matrix<double, VARIABLE_NUM, 1>::Zero(); // the initial guess values for the solution

    // Define variables that will be used during the iterative process
    MatA L_tri = Eigen::Matrix<double, LINEAR_EQUATION_NUM, VARIABLE_NUM>::Zero(); // the stricktly lower triangular matrix to be used in the Gauss-Seidel method
    MatA U_tri = Eigen::Matrix<double, LINEAR_EQUATION_NUM, VARIABLE_NUM>::Zero(); // the stricktly upper triangular matrix to be used in the Gauss-Seidel method 
    VecX x_temp = x_vec; // a temporary update version of the x_vec

    // Define variables to examine the convergence of the process
    MatA A_tmp = A_mat; // a temporary A matrix that has all diagonal terms to be zero
    MatA D = A_mat.diagonal().asDiagonal(); // extract the diagonal entries and construct a diagonal matrix D with the elements
    MatA D_L_U; // a matrix to determine if the algorithm will converge
    Eigen::MatrixXcd Eig_vals; // matrix that stores the eigen values of the D_L_U matrix, can be complex value
    double max_eig_norm = 0.0; // the largest eigen value norm
    int end_flag = 0; // a flag to indicate if the iterative process should be continue, 1 - ending the proces, 0 - continue

    // Starting the iterative method
    for (int i = 0; i < LINEAR_EQUATION_NUM; i++) {
        if (A_tmp(i,i) == 0) {
           end_flag = 1;
           std::cout << "The iterative process cannot be carried out on the input matrix." << std::endl;
           return; 
        }
        A_tmp(i,i) = 0;
    }

    if (method_code == 1 && end_flag != 1) { // Jacobi iteration
        D_L_U = D.inverse() * (-A_tmp);

    } else if (method_code == 2 && end_flag != 1) { // Gauss-Seidel iteration
        L_tri -= A_mat.triangularView<Eigen::StrictlyLower>();
        U_tri -= A_mat.triangularView<Eigen::StrictlyUpper>();
        D_L_U = (D - L_tri).inverse() * U_tri;

    } else {
        std::cout << "Invalid method code. Either 1 or 2." << std::endl;
        end_flag = 1; 
        return;
    }

    // compute the eigen values of the D_L_U matrix
    Eigen::EigenSolver<MatA> EigValSolver(D_L_U);
    Eig_vals = EigValSolver.eigenvalues(); // solve the eigen values
    // find out the largest eigen value norm
    for (int i = 0; i < VARIABLE_NUM; i++) { // find out the largest eigen value norm
        max_eig_norm = ( max_eig_norm > std::__complex_abs(Eig_vals(i)) ) ? max_eig_norm : (std::__complex_abs(Eig_vals(i)));
    }

    // note that the spectral radius is the largest eigen value norm
    // The algorithm will not converge if the spectral radius is not smaller than 1.0
    // See Theorem 7.17 in Burden_2011_Numerical Analysis_Ed9_Sec. 7.2
    if (max_eig_norm >= 1.0) {
        std::cout << "The iterative algorithm does not converge!" << std::endl;
        end_flag = 1;
        return;
    }

    // if all the tests above are passed, start the iterative procedure
    int iter = 0; // a counter of the number of iterations
    while (end_flag != 1) {
        
        // find the new x_vec and keep track on the infinity norm of the x_vec and x_temp
        double max_diff = -99999.9;

        // vectorized computation of new x, reference:
        // Burden_2011_Numerical Analysis_Ed9_Sec. 7.3
        if (method_code == 1) { // Jacobi iterative
            x_temp = D_L_U * x_vec + D.inverse() * b_vec;
        } else if (method_code == 2) { // Gauss-Seidel iterative
            x_temp = D_L_U * x_vec + (D - L_tri).inverse() * b_vec;
        }

        // compute the infinity norm between x_temp and x_vec
        for (int i = 0; i < VARIABLE_NUM; i++) {
            max_diff = std::fabs(x_temp(i) - x_vec(i)) > max_diff ? std::fabs(x_temp(i) - x_vec(i)) : max_diff;
        }

        // check if the infinity norm of the update has fallen below the accuray limit
        if (max_diff < accuracy_level && max_diff > 0.0) {
            std::cout << "The targeted accuracy level " << accuracy_level << " has been reach at the " << iter+1 << " th iteration." << std::endl;
            end_flag = 1;
            break;
        } else {
            x_vec = x_temp; // update the x_vec to the current estimate
        }

        // check if the number of iterations has reached the max number of iterations
        if (++iter >= num_iterations) {
            end_flag = 1;
            std::cout << "The maximum number of iteration reached. Ending the iterative process." << std::endl;
            return;
        }
    }
} // end of the iterativeProcess
#endif