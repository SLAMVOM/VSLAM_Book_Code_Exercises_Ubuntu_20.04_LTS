#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

// The detailed explaination of the linear triangulation method refers to
// Hartley and Zisserman (2004) Multiple View Geometry in Computer Vision
// Section 12.2 and Appendix A5.3
/**
* linear triangulation with SVD
* @param poses          poses,
* @param points         points in normalized plane
* @param pt_world       triangulated point in the world
* @return true if success
**/
inline bool triangulation(const std::vector<SE3> &poses,
                    const std::vector<Vec3> points, Vec3 &pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0); // construct the row of A mat related to the point's x coor. in the ith cam. frame's norm img plane
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1); // construct the row of A mat related to the point's y coor. in the ith cam. frame's norm img plane
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // if the ratio between the last and the second last singular values
        // is smaller than a certain threshold, the solution is accepted
        // otherwise, the quality of the solution is low, abandon the solution
            return true;
    }
    return false;
}


// converters - from cv to Eigen
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}   // namespace myslam

#endif  // MYSLAM_ALGORITHM_H