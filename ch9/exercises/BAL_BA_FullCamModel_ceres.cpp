/*****************************************************************
* This code is to implmenet the BA problem in Ch9 of VSLAM book
* but with a more elaborated camera model as in Ch 5 of the book,
* including fx, fy, p1, p2, k1, k2.
*
* Reference: Section 9.4 in VSLAM book 2nd ed.
*
* The solver used in this problem is Ceres.
*
*
* Created by: MT
* Creation Date: 2022-May-01
* Previous Edit: 2022-May-01
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
    bal_problem.WriteToPLYFile("ceresfinal.ply");

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
        CamProjectionWithFullCamModel(camera, point, predictions); // reproject the 3D point to pixel coordinates through the extrinsics and intrinsics
        // Here uses the predicted - observed value as residual terms
        // Note in the BAL dataset, the observations are given by setting the origin of the image as the center of the image.
        // Therefore, there is no c_u and c_v parameters. Also, the x-axis points to right and y-axis points up, meaning that
        // the image plane is assumed behind the camera center, i.e., the world is at the negative z direction of the camera optical center.
        // This can be thought of as using a right-hand rule notation with z-axis points to the back, or equivalently left-hand rule z-axis points to the front.
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 12 dims array
    // [0-2] : angle-axis rotation, axis is the vec after normalization, angle is the norm
    // [3-5] : translation vector
    // [6-7] : camera parameter, [6] fx focal length, [7] fy focal length
    // [8-9] : p1, p2 for tangential distortion
    // [10-11] the second and the fourth order radial distortion, k1, k2
    // point : 3D location of a single point represented in the camera frame (nonnormalized coordinates)
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithFullCamModel(const T *camera, const T *point, T *predictions) {
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

        // Apply the distrotion correction
        const T &fx   = camera[6];  // fx, focal length in the x direction, also refers to as fu
        const T &fy   = camera[7];  // fy, focal length in the y direction, also refers to as fv
        const T &tan1 = camera[8];  // p1, tangential parameter 1
        const T &tan2 = camera[9];  // p2, tangential parameter 2
        const T &rad1 = camera[10]; // k1, radial parameter 1
        const T &rad2 = camera[11]; // k2, radial parameter 2

        T XY = xp * yp;
        T r2 = xp * xp + yp * yp;
        T term1 = T(1.0) + r2 * (rad1 + rad2 * r2);

        // calculate the distorted pixel coordinates
        T X_distorted = xp * term1 + T(2.0) * tan1 * XY + tan2 * (r2 * T(2.0) * xp * xp);
        T Y_distorted = yp * term1 + tan1 * (r2 + T(2.0) * yp * yp) + T(2.0) * tan2 * XY;

        // Note: no c_x and c_y needed in this case, because those are subtracted already in the observations
        predictions[0] = fx * X_distorted;
        predictions[1] = fy * Y_distorted;

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 12, 3>( // 9D for camera params (6 for extrinsics, 3 for calibrations) + 3D for landmark pts
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

    // Note that, the data given by the BAL dataset file only contains information
    // about 6D camera poses, and camera intrinsics of f, k1, k2. However, in this
    // problem, we want to consider the 6D poses and fx, fy, p1, p2, k1, and k2.
    // Therefore, we will assume fx = fy = f, and p1 = p2 = 0 at the beginning and
    // let the system to adjust their values during the optimization.
    // Initially from the data: 9D, sequentially: 3D axis-angle + 3D translation + 1D f     + 2D k1,k2
    // Each set of cam params: 12D, sequentially: 3D axis-angle + 3D translation + 2D fx,fy + 2D p1,p2 + 2D k1,k2
    double * camera_params_ext = new double [12 * N_cam]; // create a new double array for the extended camera parameters
    
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
        double *camera_9d = cameras + camera_block_size * bal_problem.camera_index()[i]; // params to be optimized
        double *point = points + point_block_size * bal_problem.point_index()[i]; // params to be optimized

        // Allocate and Transfer the camera parameters from the original 9D cam param blocks to the 12D cam param blocks
        // Note, these are initial values, the values will be optimized during the optimizations
        double *camera_12d = camera_params_ext + 12 * bal_problem.camera_index()[i]; // each extended cam param block has 12D
        for (int j_idx = 0; j_idx < 6; j_idx++) {
            camera_12d[j_idx] = camera_9d[j_idx]; // the first 6D are the same
        }
        camera_12d[6] = camera_9d[6];  // fx = f
        camera_12d[7] = camera_9d[6];  // fy = f
        camera_12d[8] = 0.0;           // p1 = 0.0
        camera_12d[9] = 0.0;           // p2 = 0.0
        camera_12d[10] = camera_9d[7]; // k1 = k1
        camera_12d[11] = camera_9d[8]; // k2 = k2

        // Adding residual block to the ceres problem
        problem.AddResidualBlock(cost_function, loss_function, camera_12d, point);
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
    options.max_num_iterations = 200; // Ceres' default: 50 
    options.minimizer_progress_to_stdout = true; // print out some progress information
    options.num_threads = 1; // Ceres' default: 1 - number of threads used by Ceres to evaluate the Jacobian
    options.max_solver_time_in_seconds = 1e6; // Ceres' default: 1e6 - max amount of time for the solver to run

    //// Parameter block ordering options
    /*******************************************************************************
    // Some notes from Ceres:
    // "Bundle adjustment problems have a sparsity structure that makes
    // them amenable to more specialized and much more efficient
    // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
    // ITERATIVE_SCHUR solvers make use of this specialized
    // structure."
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
    int inv_marginal_portion = 1; // (1/inv_marginal_portion) portion of the points will be marginalized, set to 1 to marginalize all the points at once
    for (int i = 0; i < N_pts; i++) {
        if (i % inv_marginal_portion == 0 && inv_marginal_portion > 1) {
            ordering->AddElementToGroup(points + point_block_size * i, 1);
        } else {
            ordering->AddElementToGroup(points + point_block_size * i, 0);
        }
    }

    // Cameras second
    for (int i = 0; i < N_cam; i++) {
        ordering->AddElementToGroup(camera_params_ext + 12 * i, 1);
    }
    options.linear_solver_ordering.reset(ordering);


    // Start solving the problem
    std::cout << "Solving veres BA ... " << std::endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // After optimization,allocate back the camera model values to the original param array
    for (int i = 0; i < N_cam; i++) {
        double *camera_9d  = cameras + camera_block_size * i; // params to be optimized
        double *camera_12d = camera_params_ext + 12 * i;//

        for (int j = 0; j < 6; j++){
            camera_9d[j] = camera_12d[j]; // the first 6D are the same
        }
        camera_9d[6] = (camera_12d[6] + camera_12d[7]) / 2.0; // f = (fx + fy) / 2.0; though this may not be the best way
        camera_9d[7] = camera_12d[10]; // k1
        camera_9d[8] = camera_12d[11]; // k2
    }

    // The following code just to print out the first several camera parameters to ensure
    // the calibration values are indeed changed, one can comment the follows
    for (int i = 0; i < 5; i++) {
        double *camera_12d = camera_params_ext + 12 * i;//

        for (int j = 0; j < 12; j++){
            std::cout << camera_12d[j] << " ";
        }
        std::cout << '\n';

    }



}



