#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

// forward declarations
struct MapPoint;    // defined in mappoint.h
struct Feature;     // defined in feature.h

/**
* frame
* every frame is assigned with a unique ID, keyframe is assigned with a key frame ID
**/
struct Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;              // id of this frame
        unsigned long keyframe_id_ = 0;     // id of key frame
        bool is_keyframe_ = false;          // whether it is a keyframe
        double time_stamp_;                 // time stamp of the frame, not used here
        SE3 pose_;                          // Tcw pose - transformation from world to this camera frame
        std::mutex pose_mutex_;             // Pose data lock, the mutex class is a synchronization primitive that can be used to protect shared data from being simultaneously accessed by multiple threads
        cv::Mat left_img_, right_img_;      // stereo images

        // extracted features in the left image
        std::vector<std::shared_ptr<Feature>> features_left_;
        // corresponding features in the right image, set to nullptr if no correspondence
        std::vector<std::shared_ptr<Feature>> features_right_;

    public: // data members
        Frame() {}

        Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
              const Mat &right);

        // set and get pose, thread safe
        SE3 Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose) {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// Set keyframe and assign keyframe id
        void SetKeyFrame();

        /// factory creation mode, assign id
        static std::shared_ptr<Frame> CreateFrame();
};

}   // namespace myslam

#endif  // MYSLAM_FRAME_H