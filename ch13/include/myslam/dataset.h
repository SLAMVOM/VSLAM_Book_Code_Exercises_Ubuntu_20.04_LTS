#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam {

/**
* loading dataset
* pass in config file path when construction, using the dataset_dir in the config file as the dataset file path
* camera and the next frame image is obtained after Init
**/
class Dataset {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Dataset> Ptr;
        Dataset(const std::string& dataset_path);

        /// initialization, return a boolean indicating whether init is successful
        bool Init();

        /// create and return the next frame containing the stereo images
        Frame::Ptr NextFrame();

        /// get camera by id
        Camera::Ptr GetCamera(int camera_id) const {
            return cameras_.at(camera_id);
        }
    
    private:
        std::string dataset_path_;
        int current_image_index_ = 0;

        std::vector<Camera::Ptr> cameras_;
};
}   // namespace myslam

#endif