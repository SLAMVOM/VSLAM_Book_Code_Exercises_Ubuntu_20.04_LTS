#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;  // defined in map.h

/**
 * Backend
 * Owning a independent thread, starts optimization when Map got updated
 * Map update is triggered by the frontend
 */
class Backend {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Backend> Ptr;

        /// construction function, starts backend optimization and put it on hold
        Backend();

        // set the left and right cameras, for obtaining intrinsics and extrinsics
        void SetCameras(Camera::Ptr left, Camera::Ptr right) {
            cam_left_ = left;
            cam_right_ = right;
        }

        /// set map
        void SetMap(std::shared_ptr<Map> map) { map_ = map; }

        /// initiate map update, start optimization
        void UpdateMap();

        /// close backend thread
        void Stop();

    private:
        /// backend thread
        void BackendLoop();

        /// Optimize with respect to the given keyframes and landmarks
        void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

        std::shared_ptr<Map> map_;
        std::thread backend_thread_;
        std::mutex data_mutex_;

        std::condition_variable map_update_;
        std::atomic<bool> backend_running_;

        Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}   // namespace myslam

#endif  // MYSLAM_BACKEND_H