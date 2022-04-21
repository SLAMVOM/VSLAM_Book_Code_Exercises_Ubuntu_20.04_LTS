#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    /* This code is to solve for the example shown in Sec. 3.6.2 */
    
    cout.precision(4);
    
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();

    cout << "The unit quaternion q1 is " << q1.coeffs().transpose() << endl;
    cout << "The unit quaternion q2 is " << q2.coeffs().transpose() << endl;

    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
    Vector3d p1(0.5, 0, 0.2);

    Isometry3d T1w(q1), T2w(q2);
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);

    cout << "The transformation matrix based on unit quaternion q1 is \n" << T1w.matrix() << endl;
    cout << "The transformation matrix based on unit quaternion q2 is \n" << T2w.matrix() << endl;

    cout << "The inverse of transformation matrix T1 is \n" << T1w.inverse().matrix() << endl;

    Vector3d p2 = T2w * T1w.inverse() * p1;
    cout << endl << p2.transpose() << endl;

    cout << "\n ==================== Separate Rotation and Translation ==================== \n" << endl;

    Vector3d v1(0.0, 0.0, 0.0), v2(0.0, 0.0, 0.0);
    for (int i=0; i<3; i++) {
        v1(i) = q1.coeffs()(i);
        v2(i) = q2.coeffs()(i);
    }
    double s1, s2;
    s1 = q1.coeffs()(3);
    s2 = q2.coeffs()(3);

    Matrix3d skew1 = Matrix3d::Zero();
    skew1 << 0.0d, -v1(2), v1(1), 
             v1(2), 0.0d, -v1(0), 
             -v1(1), v1(0), 0.0d;

    Matrix3d skew2 = Matrix3d::Zero();
    skew2 << 0.0, -v2(2), v2(1), 
             v2(2), 0.0, -v2(0), 
             -v2(1), v2(0), 0.0;

    Matrix3d Mat1 = Matrix3d::Zero();
    Mat1 = v1 * v1.transpose() + s1 * s1 * Matrix3d::Identity() + 2.0 * s1 * skew1 + skew1 * skew1;

    Matrix3d Mat2 = Matrix3d::Zero();
    Mat2 = v2 * v2.transpose() + s2 * s2 * Matrix3d::Identity() + 2.0 * s2 * skew2 + skew2 * skew2;

    cout << "Rotation Matrix Mat1: \n" << Mat1 << endl;
    cout << "\n Rotation Matrix Mat1: \n" << Mat2 << endl;

    Vector3d v_trans = Vector3d::Zero();

    v_trans = Mat2 * ((Mat1.transpose() * p1) + t1 - t2);

    cout << "\n point coordinate in Frame 2:" << endl << p2.transpose() << endl;

    return 0;

}