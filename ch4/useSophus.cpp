#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

// This program demonstrates the basic usage of Sophus
int main(int argc, char** argv){
    // rotating 90 degrees along the z axis (Rotation matrix)
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
    // or using quaternion
    Quaterniond q(R);
    Sophus::SO3d SO3_R(R); // Sophus::SO3d can be constructed directly through rotation matrix
    Sophus::SO3d SO3_q(q); // Sophus::SO3d can also be constructed through quaternion
    
    // The above two methods are equivalent
    cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
    cout << "they are equal" << endl;

    // Use logarithmic map to get the Lie algebra
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    // variable hat is from vector to skew-symmetric matrix
    cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
    // respectively, .vee method is from matrix to vector, i.e., the inverse operation of hat
    cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

    // update by using the perturbation model
    Vector3d update_so3(1e-4, 0, 0); // assume a small perturbation for rotation update
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R; // left multiply, 3x3
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

    cout << "****************************************" << endl;
    // Similarly, for SE(3)
    Vector3d t(1, 0, 0); // translation 1 along the X axis
    Sophus::SE3d SE3_Rt(R, t); // construct SE3 from R and t
    Sophus::SE3d SE3_qt(q, t); // construct SE3 from q and t
    cout << "SE3 from R, t= \n" << SE3_Rt.matrix() << endl;
    cout << "SE3 from q, t= \n" << SE3_qt.matrix() << endl;
    // Lie Algebra is 6d vector, using a typedef to make the following easier
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;
    // By observing the output from Sophus, the translation is printed at front, followed by the rotation
    // Similarly, there exist hat and vee operators
    cout << "se3 hat =\n" << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // Finally, perform a transformation update
    Vector6d update_se3; // a small perturbation
    update_se3.setZero();
    update_se3(0, 0) = 1e-4d;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

    return 0;


}