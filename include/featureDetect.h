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

#include <Eigen/Core>

#include <include/3DTransformation.h>
namespace poseEstimate
{
    class feature: public pose{
    public:
        /**\brief Detect image ORB feature.
         * \param[in] rgb_image current rgb image
         * \param[out] keyPoints detect rgb key corner points
         * \param[out] descriptor the descriptor of corner points
         * */
        void orbFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);

        /**\brief Detect image SIFT feature.
         * \param[in] rgb_image current rgb image
         * \param[out] keyPoints detect rgb key corner points
         * \param[out] descriptor the descriptor of corner points
         * */
        void siftFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);

        /**\brief Detect image SURF feature.
         * \param[in] rgb_image current rgb image
         * \param[out] keyPoints detect rgb key corner points
         * \param[out] descriptor the descriptor of corner points
         * */
        void surfFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);

        /**\brief Detect image FAST feature.
         * \param[in] rgb_image current rgb image
         * \param[out] keyPoints detect rgb key corner points
         * \param[out] descriptor the descriptor of corner points
         * */
        void fastFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptor);

        //aruco for pose estimation
        /**\brief Read camera parameters and program parameters.
         * */
        void readParams(std::string cam_param);

        /**\brief Create ARTag board using arucoCornersX and arucoCornersY from [readParams] function.
         * */
        void createBoard();

        /**\brief Detect aruco corners and estimate current artag board pose.
         * \param[in] rgb_image current rgb image
         * \param[in] depth_image current depth image
         * \param[out] ids corners id number
         * \param[out] corners artag corners
         * \param[out] trans current artag board pose
         * */
        bool arucoDetect(cv::Mat rgb_image, cv::Mat depth_image, std::vector<int> ids, std::vector<std::vector<cv::Point3f>>& corners, Eigen::Matrix4f& trans);

        /**\brief Detect aruco corners and estimate current artag board pose.
         * \param[in] desc_1 current image descriptors
         * \param[in] desc_2 previous image descriptors
         * \param[out] matches feature match descriptors
         * */
        void featureMatch(cv::Mat& desc_1, cv::Mat& desc_2, std::vector<cv::DMatch>& matches);
    private:
        /**\brief Camera intrinsic parameters.*/
        cv::Mat camera_matrics_;

        /**\brief Camera model distortion.*/
        cv::Mat camera_dist_coffs_;

        /**\brief Artag board.*/
        cv::Ptr<cv::aruco::Board> board_ptr_;

        /**\brief Board image.*/
        cv::Mat board_image_;

        /**\brief Artag board corners coordinate in x axis and y axis.*/
        cv::Mat corners_x_, corners_y_;

        /**\brief Artag board length in real world.*/
        double length_;

        /**\brief Capture image ceiling.*/
        int total_capture_frame_;

        /**\brief Detect key frame distance threshold.*/
        double distance_tresh_;

        /**\brief Artag board dictionary.*/
        cv::Ptr<cv::aruco::Dictionary> dictionary_;


        /**\brief Artag board corners coordinate.*/
        std::vector<std::vector<cv::Point3f>> marker_corner_vec_;

        /**\brief Artag board corners id.*/
        std::vector<int> ids_vec_;
    };
}

#endif
