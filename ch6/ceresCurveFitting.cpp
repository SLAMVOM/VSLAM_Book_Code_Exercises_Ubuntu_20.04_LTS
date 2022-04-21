#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// the calculation model of the cost function
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // implement operator () to compute the error
    template<typename T>
    bool operator() (
        const T *const abc, // the estimated variables, 3D vector
        T *residual) const {
        // y - exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
        }
    
    const double _x, _y; // x, y data
};

int main (int argc, char **argv) {
    // same model input as before
    double ar = 1.0, br = 2.0, cr = 1.0; // ground-truth parameters values
    double ae = 2.0, be = -1.0, ce= 5.0; // initial guess of parameters
    int N = 100; // number of data points used to fit the curve
    double w_sigma = 1.0; // sigma of the noise, stdev of a standard Normal
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // random number generator

    vector<double> x_data, y_data; // the input data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));  
    }

    double abc[3] = {ae, be, ce};

    //construct the problem in ceres
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        problem.AddResidualBlock( // add the ith residual into the problem
            // using auto-diff, template params: residual type, output dimension, input dimension
            // should be the same as the struct witten before
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>( // 1 -> scalar is the dim of the cost; 3 -> the dim of the params
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            nullptr, // kernel function, leave it empty for now
            abc // estimated variables
        );
    }

    // set the solver options
    ceres::Solver::Options options; // there are lots of params that can be tuned, see doc.
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // using cholesky decomposition to solve the normal system
    options.minimizer_progress_to_stdout = true; // print to cout, a verbose option

    ceres::Solver::Summary summary; // the summary info. of the optimization process
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary); // perform the optimization
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // obtain the output
    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = ";
    for (auto a:abc) cout << a << " ";
    cout << endl;

    return 0;
}