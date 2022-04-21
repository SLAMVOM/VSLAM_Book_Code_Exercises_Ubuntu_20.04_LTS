/************************************************************************************************************************
* This script performs a curve-fitting task using the Levenberg-Marquardt (LM) method with vectorization
* The model function used is the same as the one used in Sec. 6.3 of VSLAM book by Gao Xiang (Ed2, Chinese ver.)
* Reference: Gavin (2020) "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems"
* Author: MT
* Creation date: 2022-April-07
* Previous edit: 2022-April-07
************************************************************************************************************************/

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
//using namespace Eigen;

const int method = 2; // 1 - "Levenberg"; 2 - "Quadratic"; as in Gavin (2020)

int main(int argc, char **argv) {
    cout.precision(7);

    double ar = 1.0, br = 2.0, cr = 1.0; //ground-truth values
    double ae = 2.0, be = -1.0, ce= 5.0; // initial guess
    const int N = 5000; // number of data points used during the curve fitting
    double w_sigma = 1.0; // sigma of the noise, stdev of a standard Gaussian
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // random number generator

    // user-defined LM hyperparameters
    double L_upper = 11.0;
    double L_lower = 9.0;
    double lambda_init = 1e-2; // initial value of the L-M damping parameter
    double epsilon = 1e-1; // update acceptance threshold
    Eigen::Matrix3d D;


    vector<double> x_data, y_data; // the data
    for (int i = 0; i < N; i++) {
        double x = (i * 1.0) / (N * 1.0); // IMPORTANT: notice the 100.0 is not 100
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // start Levenberg-Marquardt iterations
    int iterations = 100;
    double cost = 0, lastCost = 0;
    double lambda = lambda_init;
    double rho, alpha;
    double a_tmp, b_tmp, c_tmp; // temporary values for the model parameters
    double cost_tmp = 0.0;

    Eigen::VectorXd x_d = Eigen::Map<Eigen::VectorXd>(x_data.data(), x_data.size()); // convert std::vec to Eigen::vec
    Eigen::VectorXd y_d = Eigen::Map<Eigen::VectorXd>(y_data.data(), y_data.size()); // convert std::vec to Eigen::vec

    Eigen::Vector3d b = Eigen::Vector3d::Zero(); // bias/error terms
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); // Hessian = J^T W^{-1} J

    Eigen::Matrix<double, 3, N> J; // note: N here is a const int
    Eigen::Matrix<double, Eigen::Dynamic, 1> err;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; iter++) {

        cost = 0;
        
        // Note: In the following code, the left is Eigen::matrix, the right is Eigen::array,
        // Eigen will convert automatically in assignment
        // err: [N x 1]; J: [N x 3]
        err = y_d.array() - exp(ae * x_d.array() * x_d.array() + be * x_d.array() + ce); // [N x 1] matrix
        J.row(0) = -x_d.array() * x_d.array() * exp(ae * x_d.array() * x_d.array() + be * x_d.array() + ce); // de/da
        J.row(1) = -x_d.array() * exp(ae * x_d.array() * x_d.array() + be * x_d.array() + ce); // de/db
        J.row(2) = -exp(ae * x_d.array() * x_d.array() + be * x_d.array() + ce); // de/dc

        H = inv_sigma * inv_sigma * J * J.transpose(); // [3x3]
        b = -inv_sigma * inv_sigma * J * err; // [3x1]

        cost = err.transpose() * err; // [#]

        // L-M steps
        if (method == 1) { // 1 for "Levenberg" update method
            Eigen::Matrix3d D = H.diagonal().asDiagonal(); // construct the diagonal matrix D
            H += lambda * D; // [3x3]
        }
        else if (method == 2) { // 2 for "Quadratic" update method
            H += lambda * Eigen::Matrix3d::Identity(); // [3x3]
        } else {
            cout << "Invalid update method code: " << method << std::endl;
            break;
        }

        // solve Hx = b
        Eigen::Vector3d dx = H.ldlt().solve(b); // using the robust cholesky decomposition with pivoting
        // determine if the update contain NaN
        if ( isnan(dx[0]) ) {
            cout << "result is nan!" << endl;
            break;
        }

        // temporarily udpate the model parameter and compute rho
        a_tmp = ae + dx[0];
        b_tmp = be + dx[1];
        c_tmp = ce + dx[2];

        // L-M - computing the rho value
        /// first compute the cost of the updated value
        cost_tmp = 0.0;
        err = y_d.array() - exp(a_tmp * x_d.array() * x_d.array() + b_tmp * x_d.array() + c_tmp); // [N x 1] matrix
        cost_tmp = err.transpose() * err; // [#]

        if (method == 1) { // 1 for "Levenberg" update method
            // compute the rho value for the Levenberg method
            rho = (cost - cost_tmp) / (dx.transpose() * (lambda * D * dx + b)); // [#]
            if (rho > epsilon) {
                ae = a_tmp;
                be = b_tmp;
                ce = c_tmp;
                lambda = std::max(lambda / L_lower, 1e-7);
            } else {
                lambda = std::min(lambda * L_upper, 1e7);
            }
        }
        else if (method == 2) { // 2 for "Quadratic" update method
            // compute the alpha value for the Quadratic method 
            alpha = (b.transpose() * dx)[0] / ((cost_tmp - cost)/2 + 2 * ((b.transpose() * dx)[0])); // [#]
            Eigen::Vector3d dx_tmp = dx * alpha;
            a_tmp = ae + dx_tmp[0];
            b_tmp = be + dx_tmp[1];
            c_tmp = ce + dx_tmp[2];
            
            // evaluate the model with the updated parameter values again
            cost_tmp = 0.0; // reset temporary cost
            err = y_d.array() - exp(a_tmp * x_d.array() * x_d.array() + b_tmp * x_d.array() + c_tmp); // [N x 1] matrix
            cost_tmp = err.transpose() * err; // [#]

            // compuate the rho value for the Quadratic update method
            rho = (cost - cost_tmp) / (dx.transpose() * (lambda * dx_tmp + b)); // [#]
            if (rho > epsilon) {
                ae = a_tmp;
                be = b_tmp;
                ce = c_tmp;
                lambda = std::max(lambda / (1+alpha), 1e-7);
            } else {
                lambda = lambda + std::abs((cost_tmp - cost) / 2 / alpha);
            }
        } 

        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << " >= last cost: " << lastCost << ", break." << endl;
            break;
        }

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;    
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;

    return 0; 

}