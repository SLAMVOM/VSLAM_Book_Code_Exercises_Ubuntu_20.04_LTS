#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

// vertex of the curve model, template params: the parameter dimensionality and data type
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // override the reset function
        virtual void setToOriginImpl() override {
            _estimate << 0, 0, 0;
        }

        // override the plus operator, just plain vector addition
        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Vector3d(update);
        }

        // read and write functions - leaving them to be empty
        virtual bool read(istream &in) {}
        virtual bool write(ostream &out) const {}
};

// error/bias model, template params: dim of the observed values, data type, vertex's data type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

        // compute the error of the fitted model
        virtual void computeError() override{
            const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
            const Eigen::Vector3d abc = v->estimate(); // the estimate() method and _estimate member are in "g2o/core/base_vertex.h"
            _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
        }

        // compute the Jaconbian matrix
        virtual void linearizeOplus() override {
            const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
            const Eigen::Vector3d abc = v->estimate();
            double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
            _jacobianOplusXi[0] = -_x * _x * y; // the _jacobianOplusXi member here is inherited from BaseBinaryEdge class
            _jacobianOplusXi[1] = -_x * y;
            _jacobianOplusXi[2] = -y;
        }

        virtual bool read(istream &in) {}
        virtual bool write(ostream &out) const {}

    public:
        double _x; // the value of x; note that y is given in _measurement
};


int main (int argc, char **argv){
    double ar = 1.0, br = 2.0, cr = 1.0; // ground-truth value
    double ae = 2.0, be = -1.0, ce= 5.0; // initial guess of params
    int N = 100; // number of data points
    double w_sigma = 1.0; // stdev of the noise for a standard Normal
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; //rnadom number generator

    vector<double> x_data, y_data; // the input data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // construct graph optimization, first setting g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType; // the parameters have dim of 3, error is a scalar with dim 1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // using a linear solver

    // choose an optimization method from one of Gauss-Newton, Levenberg-Marquardt, or DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // set the solver algorithm
    optimizer.setVerbose(true); // set to print the results

    // add vertex into the graph
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce)); // the setEstimate() method can be found at: "g2o/core/base_vertex.h"
    v->setId(0); // the _id member is under "g2o/core/parameter.h", and the setId() method is defined at "g2o/core/optimizable_graph.h"
    optimizer.addVertex(v);

    // add edges into the graph
    for (int i = 0; i < N; i++) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]); // Note: the new operator returns a unique pointer to the object
        edge->setId(i);
        edge->setVertex(0, v); // set the connection to the vertex to be optimized, the setVertex() method is defined in "g2o/core/hyper_graph.h"
        edge->setMeasurement(y_data[i]); // edges are measurements, the _measurement member is inherited from BaseEdge class defined in "g2o/core/base_edge.h"
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // set the information matrix as the inverse covariance
        optimizer.addEdge(edge);
    }

    // Carry out the optimization
    cout << "Start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // print the results
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}
