#ifndef _FEATURE_DETECT_H_
#define _FEATURE_DETECT_H_

#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
//#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>

#include <include/3DTransformation.h>
namespace poseEstimate
{
class feature: public pose{
public:
    //
    void orbFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);
    void siftFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);
    void surfFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);
    void fastFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);

    //aruco for pose estimation
    void readParams(std::string cam_param);
    void createBoard();
    bool arucoDetect(cv::Mat rgb_image, cv::Mat depth_image, std::vector<int> ids, std::vector<std::vector<cv::Point3f>>& corners, Eigen::Matrix4f& trans);

    //
    void featureMatch(cv::Mat& desc_1, cv::Mat& desc_2, std::vector<cv::DMatch>& matches);
private:
    cv::Mat camera_matrics_;
    cv::Mat camera_dist_coffs_;
    cv::Ptr<cv::aruco::Board> board_ptr_;
    cv::Mat board_image_;
    cv::Mat corners_x_, corners_y_;
    double length_;
    cv::Vec3d cur_rvec_;
    cv::Vec3d cur_tvec_;

    int total_capture_frame_;
    int capture_frame_num_;

    double distance_tresh_;

    std::vector<int> params;

    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    std::vector<std::vector<cv::Point3f>> marker_corner_vec_;
    std::vector<int> ids_vec_;



};
}

#endif
