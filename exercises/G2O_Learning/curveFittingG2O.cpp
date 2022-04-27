/**************************************************************************************
This is a exercise to use the G2O package to perform a curve fitting task
using the same function model and input data from the Ceres curve fitting example from:
https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/curve_fitting.cc

Author: MT
Creation Date: 2022-April-25
Previous Edit: 2022-April-25
**************************************************************************************/

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;


// Data generated using the following octave code.
//   randn('seed', 23497);
//   m = 0.3;
//   c = 0.1;
//   x=[0:0.075:5];
//   y = exp(m * x + c);
//   noise = randn(size(x)) * 0.2;
//   y_observed = y + noise;
//   data = [x', y_observed'];

const int kNumObservations = 67;
// clang-format off
const double data[] = {
  0.000000e+00, 1.133898e+00,
  7.500000e-02, 1.334902e+00,
  1.500000e-01, 1.213546e+00,
  2.250000e-01, 1.252016e+00,
  3.000000e-01, 1.392265e+00,
  3.750000e-01, 1.314458e+00,
  4.500000e-01, 1.472541e+00,
  5.250000e-01, 1.536218e+00,
  6.000000e-01, 1.355679e+00,
  6.750000e-01, 1.463566e+00,
  7.500000e-01, 1.490201e+00,
  8.250000e-01, 1.658699e+00,
  9.000000e-01, 1.067574e+00,
  9.750000e-01, 1.464629e+00,
  1.050000e+00, 1.402653e+00,
  1.125000e+00, 1.713141e+00,
  1.200000e+00, 1.527021e+00,
  1.275000e+00, 1.702632e+00,
  1.350000e+00, 1.423899e+00,
  1.425000e+00, 1.543078e+00,
  1.500000e+00, 1.664015e+00,
  1.575000e+00, 1.732484e+00,
  1.650000e+00, 1.543296e+00,
  1.725000e+00, 1.959523e+00,
  1.800000e+00, 1.685132e+00,
  1.875000e+00, 1.951791e+00,
  1.950000e+00, 2.095346e+00,
  2.025000e+00, 2.361460e+00,
  2.100000e+00, 2.169119e+00,
  2.175000e+00, 2.061745e+00,
  2.250000e+00, 2.178641e+00,
  2.325000e+00, 2.104346e+00,
  2.400000e+00, 2.584470e+00,
  2.475000e+00, 1.914158e+00,
  2.550000e+00, 2.368375e+00,
  2.625000e+00, 2.686125e+00,
  2.700000e+00, 2.712395e+00,
  2.775000e+00, 2.499511e+00,
  2.850000e+00, 2.558897e+00,
  2.925000e+00, 2.309154e+00,
  3.000000e+00, 2.869503e+00,
  3.075000e+00, 3.116645e+00,
  3.150000e+00, 3.094907e+00,
  3.225000e+00, 2.471759e+00,
  3.300000e+00, 3.017131e+00,
  3.375000e+00, 3.232381e+00,
  3.450000e+00, 2.944596e+00,
  3.525000e+00, 3.385343e+00,
  3.600000e+00, 3.199826e+00,
  3.675000e+00, 3.423039e+00,
  3.750000e+00, 3.621552e+00,
  3.825000e+00, 3.559255e+00,
  3.900000e+00, 3.530713e+00,
  3.975000e+00, 3.561766e+00,
  4.050000e+00, 3.544574e+00,
  4.125000e+00, 3.867945e+00,
  4.200000e+00, 4.049776e+00,
  4.275000e+00, 3.885601e+00,
  4.350000e+00, 4.110505e+00,
  4.425000e+00, 4.345320e+00,
  4.500000e+00, 4.161241e+00,
  4.575000e+00, 4.363407e+00,
  4.650000e+00, 4.161576e+00,
  4.725000e+00, 4.619728e+00,
  4.800000e+00, 4.737410e+00,
  4.875000e+00, 4.727863e+00,
  4.950000e+00, 4.669206e+00,
};
// clang-format on

/************************************************************************************
The G2O workflow:
    1. Define vertex class(s)/structure(s)
    2. Define edge class(s)/structure(s)
    3. Setup and allocate a solver object (with an optimization algorithm)
    4. Construct the optimizatoin graph by adding vertices and edges to the optimizer
    5. Perform optimization and return results
************************************************************************************/

// The model function is f(x) = exp(m * x + c), parameters to be found: m and c

// 1. Define vertex class, template parameters: parameter dimensional and data type
class CurveFittingVertex : public g2o::BaseVertex<2, Eigen::Vector2d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // override the reset function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate << 0, 0; // same initial guess as in the ceres example
        }

        // override the Oplus operator to be addition to the old values, this plus symbol is \boxplus in the g2o doc
        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Vector2d(update);
        }

        // read and write functions - leaving to be empty here
        virtual bool read(std::istream &in) {return false;}
        virtual bool write(std::ostream &out) const {return false;}
};


/**
 * measurement for a point on the curve
 *
 * Here the measurement is the point which is lies on the curve.
 * The error function computes the difference between the curve
 * and the point.
 */
// 2. Define edge class, template parameters: dim of the observed values, date type, vertex's data type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {};

        // Define the computeError() method to calculate each individual error term belong to this class
        virtual void computeError() override {
            const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]); // in this case, only one set of parameters to be optimized
            const Eigen::Vector2d m_c = v->estimate();
            _error(0, 0) = _measurement - std::exp(m_c(0, 0) * _x + m_c(1, 0));
        }

        // Compute the Jacobian matrix
        virtual void linearizeOplus() override {
            const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]); // the _vertices attribute is inherited from g2o::HyperGraph::Edge defined in "g2o/core/hyper_graph.h"
            const Eigen::Vector2d m_c = v->estimate();
            double y_est = std::exp(m_c[0] * _x + m_c[1]);
            _jacobianOplusXi[0] = -_x * y_est; // de/dm_c[0] = (de/dy_est) * (dy_est/dm_c[0])
            _jacobianOplusXi[1] = -y_est; // de/dm_c[1] = (de/dy_est) * (dy_est/dm_c[1])
        }

        virtual bool read(istream &in) {return false;}
        virtual bool write(ostream &out) const {return false;}

    public: // private:
      double _x; // the value of x
};


int main (int argc, char **argv) {
    double m_truth = 0.3, c_truth = 0.1; // groundtruth values
    double w_var = 1.0; // variance of the noise as specified by the example
    double me = 0.0, ce = 0.0; // initial guess of params

    // 3. Setup and allocate a solver object (with an optimization algorithm)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<2, 1>> BlockSolverType; // the parameters have dim of 2, error is a scalar with dim 1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // using a linear solver

    // using the Levenberg-Marquadt method (or Gauss-Newton)

    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // g2o::OptimizationAlgorithmGaussNewton
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph optimizer
    optimizer.setAlgorithm(solver); // another way to set optimization algorithm can see https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/data_fitting/curve_fit.cpp#L137
    optimizer.setVerbose(true); // set to print verbose info

    // 4. Construct the optimization graph by adding vertices and edges to the optimizer
    // 4.1 adding vertex into the graph, in this case, only one parameter block
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector2d(me, ce)); // the setEstimate() method can be found at: "g2o/core/base_vertex.h"
    v->setId(0); // the _id member is under "g2o/core/parameter.h", and the setId() method is defined at "g2o/core/optimizable_graph.h"
    optimizer.addVertex(v);

    // 4.2 adding edges into the graph
    for (int i = 0; i < kNumObservations; i++) {
        CurveFittingEdge *edge = new CurveFittingEdge(data[2*i]); // Note: in this exercise, x=data[2*i]; y=data[2*i+1]
        edge->setId(i);
        edge->setVertex(0, v); // set the 0th vertex on the hyper-edge to the pointer supplied, the setVertex() method is defined in "g2o/core/hyper_graph.h"
        edge->setMeasurement(data[2*i+1]); // edges are measurements, the _measurement member is inherited from BaseEdge class defined in "g2o/core/base_edge.h"
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0/w_var); // set the information matrix as the inverse covariance
        optimizer.addEdge(edge);
    }

    // 5. Perform optimization and return results
    std::cout << "Start Optimization" << std::endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(40);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // print the results
    Eigen::Vector2d m_c_estimate = v->estimate();
    cout << "estimated model: " << m_c_estimate.transpose() << endl;

    return 0;
}
