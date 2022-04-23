#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {

/**
* @brief map
* the interaction with the map: frontend invokes InsertKeyframe and InsertMapPoint to add new frame and mappoint to the map
* backend maintains the structure of the map, identify and remove outliers etc.
**/
class Map {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType; // Unordered map is an associative container that contains key-value pairs with unique keys
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType; // Unordered map is an associative container that contains key-value pairs with unique keys

        Map() {}

        /// add one keyframe
        void InsertKeyFrame(Frame::Ptr frame);
        /// add one map point
        void InsertMapPoint(MapPoint::Ptr map_point);

        /// obtain all the mappoints
        LandmarksType GetAllMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;
        }
        /// obtain all the keyframes
        KeyframesType GetAllKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

        /// get and activate map points
        LandmarksType GetActiveMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        /// get and activate key frame
        KeyframesType GetActiveKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;
        }

        /// clean the mappoints with an observed time of zero
        void CleanMap();

        private:
        // set the old key frames to inactive status
        void RemoveOldKeyframe();

        std::mutex data_mutex_;
        LandmarksType landmarks_;               // all landmarks
        LandmarksType active_landmarks_;        // active landmarks
        KeyframesType keyframes_;               // all key-frames
        KeyframesType active_keyframes_;        // active key-frames

        Frame::Ptr current_frame_ = nullptr;

        // settings
        int num_active_keyframes_ = 7;          // number of active keyframes retained
};
} // namespace myslam

#endif  // MAP_H