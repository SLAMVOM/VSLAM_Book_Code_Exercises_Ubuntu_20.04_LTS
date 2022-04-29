/*****************************************************************
* This code is to re-implmenet the BA problem in Ch9 of VSLAM book
* using Ceres with Elimination group. One can use the Elimination
* group to achieve the effect of Schur elimination in Ceres.
*
* References: Ch9 in VSLAM book 2nd ed. and Ch10 in the 1st ed.
*
* Created by: MT
* Creation Date: 2022-April-28
* Previous Edit: 2022-April-28
*****************************************************************/


#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "rotation.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundleAdjustmentCeres bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

class ReprojectionResidual {
public:
    ReprojectionResidual(double observation_x, double observation_y) : observed_x(observation_x),
                                                                       observed_y(observation_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T *residuals) const {
        T predictions[2]; // the predicted pixel coordinates
        CamProjectionWithDistortion(camera, point, predictions); // reproject the 3D point to pixel coordinates through the extrinsics and intrinsics
        // Here uses the predicted - observed value as residual terms
        // Note in the BAL dataset, the observations are given by setting the origin of the image as the center of the image.
        // Therefore, there is no c_u and c_v parameters. Also, the x-axis points to right and y-axis points up, meaning that
        // the image plane is assumed behind the camera center, i.e., the world is at the negative z direction of the camera optical center.
        // This can be thought of as using a right-hand rule notation with z-axis points to the back, or equivalently left-hand rule z-axis points to the front.
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation, axis is the vec after normalization, angle is the norm
    // [3-5] : translation vector
    // [6-8] : camera parameter, [6] focal length, [7-8] the second and the fourth order radial distortion
    // point : 3D location of a single point represented in the camera frame (nonnormalized coordinates)
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p); // first rotation
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion (normalized image plane coordinates)
        T xp = -p[0] / p[2]; // note the negative sign is particular for the BAL convention
        T yp = -p[1] / p[2]; // note the negative sign is particular for the BAL convention

        // Apply the second and the fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);

        const T &focal = camera[6]; // here assumes the pixel is square and fx = fy
        // Note: no c_x and c_y needed in this case, because those are subtracted already in the observations
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 9, 3>( // 9D for camera params (6 for extrinsics, 3 for calibrations) + 3D for landmark pts
            new ReprojectionResidual(observed_x, observed_y))); 
    }

private:
    double observed_x;
    double observed_y;
}; // ReprojectionResidual

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();
    int N_pts = bal_problem.num_points();
    int N_cam = bal_problem.num_cameras();

    // Observations is 2 * num_observations long-type array observations
    // [u_1, u_2, ..., u_n], where each u_i is the two dimensional x
    // and y positions of the observation.
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); i++) {
        ceres::CostFunction* cost_function;

        // Each Residual block takes 2D point's x & y, intrinsics and camera poses as input
        // and outputs a 2-dimensional Residual
        cost_function = ReprojectionResidual::Create(observations[2 * i + 0], observations[2 * i + 1]);

        ceres::LossFunction *loss_function = nullptr;
        loss_function = new ceres::HuberLoss(1.0); // If enabled, using Huber's loss function.

        // Each observation corresponds to a pair of one camera and one point
        // which are identified as camera_index()[i] and point_index()[i],
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i]; // params to be optimized
        double *point = points + point_block_size * bal_problem.point_index()[i]; // params to be optimized

        // Adding residual block to the ceres problem
        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show verbose information
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    // setup the ceres solver
    // More options see: http://ceres-solver.org/nnls_solving.html
    ceres::Solver::Options options;

    //// Linear solver options
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // specify the solver type

    //// Minimizer options
    options.max_num_iterations = 50; // Ceres' default: 50 
    options.minimizer_progress_to_stdout = true; // print out some progress information
    options.num_threads = 1; // Ceres' default: 1 - number of threads used by Ceres to evaluate the Jacobian
    options.max_solver_time_in_seconds = 1e6; // Ceres' default: 1e6 - max amount of time for the solver to run

    //// Parameter block ordering options
    /*******************************************************************************
    // Some notes from Ceres:
    // Bundle adjustment problems have a sparsity structure that makes
    // them amenable to more specialized and much more efficient
    // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
    // ITERATIVE_SCHUR solvers make use of this specialized
    // structure.
    //
    // This can either be done by specifying Options::ordering_type =
    // ceres::SCHUR, in which case Ceres will automatically determine
    // the right ParameterBlock ordering, or by manually specifying a
    // suitable ordering vector and defining
    // Options::num_eliminate_blocks.
    //
    // According to Ceres: For the best performance, the elimination group
    // should be as large as possible. Fot a standard BA problem, this corresponds
    // to the first elimination group containing all the 3d points,
    // the second cotaining all the cameras parameter blocks.
    // The smaller the group index, the earlier the group will be solved
    *******************************************************************************/
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
    // Point first - AddElementToGroup(const double *element, const int groupIdx)
    int inv_marginal_portion = 4; // (1/inv_marginal_portion) portion of the points will be marginalized
    for (int i = 0; i < N_pts; i++) {
        if (i % inv_marginal_portion == 0 && inv_marginal_portion > 1) {
            ordering->AddElementToGroup(points + point_block_size * i, 1);
        } else {
            ordering->AddElementToGroup(points + point_block_size * i, 0);
        }
    }

    // Cameras second
    for (int i = 0; i < N_cam; i++) {
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
    }
    options.linear_solver_ordering.reset(ordering);


    // Start solving the problem
    std::cout << "Solving veres BA ... " << std::endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

