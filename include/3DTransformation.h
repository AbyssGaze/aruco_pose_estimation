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
        /**\brief Detect image FAST feature.
         * \param[in] src_points current frame corners coordinate
         * \param[in] tag_points previous frame corresponding corners coordinate
         * \param[out] trans using SVD to estimate adjacent frames pose
         * */
        void poseEstimation(std::vector<cv::Point3f>& src_points, std::vector<cv::Point3f>& tag_points, Eigen::Matrix4f& trans);
    };
}

#endif
