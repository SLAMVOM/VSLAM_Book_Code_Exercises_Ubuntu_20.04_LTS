#include "myslam/camera.h"

namespace myslam {

Camera::Camera() {
}

Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) { // convert world pt to camera frame
    return pose_ * T_c_w * p_w;
}

Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) { // convert pt from camera to world frame
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Vec2 Camera::camera2pixel(const Vec3 &p_c) { // from camera frame to pixel coordinates
    return Vec2(
        fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
        fy_ * p_c(1, 0) / p_c(2, 0) + cy_
    );
}

Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) { // from pixel coordinates to camera frame
    return Vec3(
        (p_p(0, 0) - cx_) * depth / fx_,
        (p_p(1, 0) - cy_) * depth / fy_,
        depth
    );
}

Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) { // from world frame to pixel coordinates
    return camera2pixel(world2camera(p_w, T_c_w));
}

Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) { // from pixel coordinates to world frame
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

}
