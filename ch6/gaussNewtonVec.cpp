#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    cout.precision(7);
    
    double ar = 1.0, br = 2.0, cr = 1.0; //ground-truth values
    double ae = 2.0, be = -1.0, ce= 5.0; // initial guess
    const int N = 5000; // number of data points used during the curve fitting
    double w_sigma = 1.0; // sigma of the noise, stdev of a standard Gaussian
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // random number generator

    vector<double> x_data, y_data; // the data
    for (int i = 0; i < N; i++) {
        double x = (i * 1.0) / (N * 1.0); // IMPORTANT: notice the 100.0 is not 100
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // start Gauss-Newton iterations
    int iterations = 100;
    double cost = 0, lastCost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    Eigen::VectorXd x_d = Eigen::Map<Eigen::VectorXd>(x_data.data(), x_data.size()); // convert std::vec to Eigen::vec
    Eigen::VectorXd y_d = Eigen::Map<Eigen::VectorXd>(y_data.data(), y_data.size()); // convert std::vec to Eigen::vec

    Vector3d b = Vector3d::Zero(); // bias/error terms
    Matrix3d H = Matrix3d::Zero(); // Hessian = J^T W^{-1} J in Gauss-Newton

    Matrix<double, 3, N> J; // note: N here is a const int
    Matrix<double, Dynamic, 1> err;

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

        // solve Hx = b
        Vector3d dx = H.ldlt().solve(b);
        if ( isnan(dx[0]) ) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << " >= last cost: " << lastCost << ", break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;

    return 0;
}