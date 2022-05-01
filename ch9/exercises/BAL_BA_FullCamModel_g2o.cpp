/*****************************************************************
* This code is to implmenet the BA problem in Ch9 of VSLAM book
* but with a more elaborated camera model as in Ch 5 of the book,
* including fx, fy, p1, p2, k1, k2.
*
* Reference: Section 9.4 in VSLAM book 2nd ed.
*
* The solver used in this problem is g2o.
*
* Created by: MT
* Creation Date: 2022-April-28
* Previous Edit: 2022-May-01
*****************************************************************/

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

// using namespace Sophus;
using namespace Eigen;
using namespace std;

/// pose and intrinsics structure
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data addresses
    // each data_addr is 12D: 3D axis-angle + 3D translation + 2D fx,fy + 2D p1,p2 + 2D k1,k2
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = Sophus::SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        fx = data_addr[6];
        fy = data_addr[7];
        p1 = data_addr[8];
        p2 = data_addr[9];
        k1 = data_addr[10];
        k2 = data_addr[11];
    }

    /// stores the estimated values into memory
    void set_to(double *data_addr) {
        auto r = rotation.log(); // angle-axis representation of rotation
        for (int i = 0; i < 3; i++) data_addr[i] = r[i];
        for (int i = 0; i < 3; i++) data_addr[i + 3] = translation[i];
        data_addr[6]  = fx;
        data_addr[7]  = fy;
        data_addr[8]  = p1;
        data_addr[9]  = p2;
        data_addr[10] = k1;
        data_addr[11] = k2;
    }

    Sophus::SO3d rotation; // the rotation is from world frame to camera frame: R_cw
    Vector3d translation = Vector3d::Zero(); // the translation is t_c^{wc}
    double fx = 0.0, fy = 0.0;
    double p1 = 0.0, p2 = 0.0;
    double k1 = 0.0, k2 = 0.0;
}; // PoseAndIntrinsics


/** The G2O workflow
 * 1. Define vertex class(es)/structure(s)
 * 2. Define edge class(es)/structure(s)
 * 3. Setup and allocate a solver object (with a specified optimization algorithm)
 * 4. Construct the optimization graph by adding vertices and edges to the optimizer
 * 5. Pefrom optimization and return results
 */


//// 1. Define vertex class(es)/structure(s)
/// Vertices for poses and intrinsics, 12D, sequentially: 3D axis-angle + 3D translation + 2D fx,fy + 2D p1,p2 + 2D k1,k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<12, PoseAndIntrinsics> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexPoseAndIntrinsics() {}

        // override the resetn function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate = PoseAndIntrinsics(); // set _estimate as a PoseAndIntrinsics object
        }

        // update rule of the estimated values
        virtual void oplusImpl(const double *update) override {
            _estimate.rotation = Sophus::SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
            _estimate.translation += Vector3d(update[3], update[4], update[5]);
            _estimate.fx += update[6];
            _estimate.fy += update[7];
            _estimate.p1 += update[8];
            _estimate.p2 += update[9];
            _estimate.k1 += update[10];
            _estimate.k2 += update[11];
        }

        /// reproject and calibrate a world point based on the estimated camera extrinsics and intrinsics to obtain the image plane coordintes
        Vector2d reproject(const Vector3d &point) {
            // p_c^{PC}: superscript ^{PC} means from Camera's optical center to Point; subscript _{c} means expressed in camera frame
            Vector3d p_cam = _estimate.rotation * point + _estimate.translation; // p_cam <=> p_c^{PC}
            p_cam = -p_cam / p_cam[2];
            // calculate several intermediate terms to ease computation - see Section 5.1.2 in VSLAM Book 2nd Ed.
            double r2 = p_cam.squaredNorm();
            double X = p_cam[0];
            double Y = p_cam[1];
            double XY = X * Y;
            // Extract data from _estimate
            double p1 = _estimate.p1;
            double p2 = _estimate.p2;
            double k1 = _estimate.k1;
            double k2 = _estimate.k2;
            double term1 = 1.0 + r2 * (k1 + k2 * r2);
            double X_distorted = X * term1 + 2 * p1 * XY + p2 * (r2 + 2 * X * X);
            double Y_distorted = Y * term1 + p1 * (r2 + 2 * Y * Y) + 2 * p2 * XY;
            return Vector2d(_estimate.fx * X_distorted,
                            _estimate.fy * Y_distorted);
        }


        virtual bool read(istream &in) {return false;}

        virtual bool write(ostream &out) const {return false;}

}; // VertexPoseAndIntrinsics


class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexPoint() {}

        // override the reset function, the setToOriginImpl() method defined in "g2o/core/optimizable_graph.h"
        virtual void setToOriginImpl() override {
            _estimate = Eigen::Vector3d(0, 0, 0);
        }

        // override the Oplus operator to be addition to the old values
        virtual void oplusImpl(const double *update) override {
            _estimate += Eigen::Vector3d( update[0], update[1], update[2] ); // coor. update of 3D landmarks is simple addition
        }

        virtual bool read(istream &in) override {return false;}
        virtual bool write(ostream &out) const override {return false;}
}; // VertexPoint


// 2. Define edge class(es)/structure(s)
// g2o::BaseBinaryEdge<D,E, VertexXi, VertexXj> where D: error dimension; E: error data type
class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // override the error computation function - Notice: this method should be changed accordingly to fit the problem setup
        virtual void computeError() override {
            auto v0 = (VertexPoseAndIntrinsics *) _vertices[0]; // the _vertices attribute is inherited from g2o::HyperGraph::Edge defined in "g2o/core/hyper_graph.h"
            auto v1 = (VertexPoint *) _vertices[1];
            auto reproj = v0->reproject(v1->estimate());
            // Note: the reprojection error is defined as estimated - measured here
            _error = reproj - _measurement; // the _error protected attribute is defined under g2o::BaseEdge in "g2o/core/base_edge.h"
        }

        // using numerical derivatives
        virtual bool read(istream &in) {return false;}
        virtual bool write(ostream &out) const {return false;}

}; // EdgeProjection

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    // Note that, the data given by the BAL dataset file only contains information
    // about 6D camera poses, and camera intrinsics of f, k1, k2. However, in this
    // problem, we want to consider the 6D poses and fx, fy, p1, p2, k1, and k2.
    // Therefore, we will assume fx = fy = f, and p1 = p2 = 0 at the beginning and
    // let the system to adjust their values during the optimization.
    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("g2ofinal.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();
    const int N_cam = bal_problem.num_cameras(); // get the number of cameras in the problem

    // 3. Allocate and setup a solver object (with a specified optimization algorithm)
    // camera dimension 12, landmark dimension 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<12, 3>> BlockSolverType; // camera 12D, landmarks 3D
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type, PoseMatrixType defined in "g2o/core/block_solver.h"
    // using one of the gradient method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // Graph model
    optimizer.setAlgorithm(solver); // Setup the solver
    optimizer.setVerbose(true);     // Turn on verbose output for debugging

    // create vectors to store the created vertices
    std::vector<VertexPoseAndIntrinsics *> vertex_cameras;
    std::vector<VertexPoint *> vertex_points;

    // 4. Construct the optimization graph by adding vertices and edges to the optimizer
    // 4.1 adding vertiices into the graph

    // build g2o problem

    // Camera vertices
    // Note that, the data given by the BAL dataset file only contains information
    // about 6D camera poses, and camera intrinsics of f, k1, k2. However, in this
    // problem, we want to consider the 6D poses and fx, fy, p1, p2, k1, and k2.
    // Therefore, we will assume fx = fy = f, and p1 = p2 = 0 at the beginning and
    // let the system to adjust their values during the optimization.
    // Initially from the data: 9D, sequentially: 3D axis-angle + 3D translation + 1D f     + 2D k1,k2
    // Each set of cam params: 12D, sequentially: 3D axis-angle + 3D translation + 2D fx,fy + 2D p1,p2 + 2D k1,k2
    double * camera_params_ext = new double [12 * N_cam]; // create a new double array for the extended camera parameters
    const double *observations = bal_problem.observations();
    for (int i = 0; i < bal_problem.num_cameras(); i++) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();

        // Allocate the appropriate camera pose and intrinsics values
        double *camera_9d  = cameras + camera_block_size * i;
        double *camera_12d = camera_params_ext + 12 * i; // each extended parameter block has 12 D
        for (int j = 0; j < 6; j++) {
            camera_12d[j] = camera_9d[j]; // first 6D are the same
        }
        camera_12d[6] = camera_9d[6];  // fx = f
        camera_12d[7] = camera_9d[6];  // fy = f
        camera_12d[8] = 0.0;           // p1 = 0.0
        camera_12d[9] = 0.0;           // p2 = 0.0
        camera_12d[10] = camera_9d[7]; // k1 = k1
        camera_12d[11] = camera_9d[8]; // k2 = k2

        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera_12d));// the setEstimate() method can be found at: "g2o/core/base_vertex.h"
        // g2o in BA needs to manually set vertices to be marginalized
        // v->setMarginalized(true); // BA in g2o needs to manually set vertices to be marginalized
        optimizer.addVertex(v);
        vertex_cameras.push_back(v);
    }

    // Landmark vertices
    for (int i = 0; i < bal_problem.num_points(); i++) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras()); // the Id values follows the last one from camera pose
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // //g2o in BA needs to manually set vertices to be marginalized
        v->setMarginalized(true); // BA in g2o needs to manually set vertices to be marginalized
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // 4.2 adding edges into the graph
    for (int i = 0; i < bal_problem.num_observations(); i++) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_cameras[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // 5. Pefrom optimization and return results
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(200);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization time used: " << time_used.count() << " seconds." << endl;

    // set to BAL problem
    double *camera_12d = new double[12];
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_cameras[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera_12d);
        // Allocate the appropriate camera pose and intrinsics values
        for (int j = 0; j < 6; j++) {
            camera[j] = camera_12d[j]; // first 6D are the same
        }
        camera[6] = (camera_12d[6] + camera_12d[7]) / 2.0; // f = (fx + fy) / 2.0; though this may not be the best way
        camera[7] = camera_12d[10]; // k1
        camera[8] = camera_12d[11]; // k2
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }


}
