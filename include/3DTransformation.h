#ifndef _3DTRANSFORMATION_H_
#define _3DTRANSFORMATION_H_

#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <vector>
#include <fstream>

namespace poseEstimate
{
class pose
{
public:
    void poseEstimation(std::vector<cv::Point3f>& src_points, std::vector<cv::Point3f>& tag_points, Eigen::Matrix4f& trans);

    void arucoInitial(const std::string fileName);

    void markerDetect(const cv::Mat rgb_mat, const cv::Mat depth_mat, std::vector<cv::Point3f>& cornersVec);

private:
    void projection(cv::Mat depth_map, cv::Point2f uv, cv::Point3f& xyz);

private:
    cv::Mat camera_matrics_;
    cv::Mat camera_dist_coffs_;
    cv::Mat corners_x_, corners_y_;
    cv::Ptr<cv::aruco::Board> board_ptr_;
    double distance_tresh_;
    int total_capture_frame_;
    std::vector<int> params;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    std::vector<std::vector<cv::Point3f>> corners_;
};
}

#endif
