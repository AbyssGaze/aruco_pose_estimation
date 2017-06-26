#include <include/3DTransformation.h>

namespace poseEstimate
{
void pose::poseEstimation(std::vector<cv::Point3f> &src_points, std::vector<cv::Point3f> &tag_points, Eigen::Matrix4f &trans)
{
    int points_num = src_points.size();
    //计算两组点的质心
    cv::Mat center_src(3, 1, CV_32F, float(0)), center_tag(3, 1, CV_32F, float(0));
//    cv::Point3f center_src = cv::Point3f(0.0, 0.0, 0.0), center_tag = cv::Point3f(0.0, 0.0, 0.0);
    for(size_t i = 0; i < src_points.size(); ++i){
        center_src.at<float>(0) += src_points[i].x;
        center_src.at<float>(1) += src_points[i].y;
        center_src.at<float>(2) += src_points[i].z;
    }
    center_src = center_src / src_points.size();

    for(size_t i = 0; i < tag_points.size(); ++i){
        center_tag.at<float>(0) += tag_points[i].x;
        center_tag.at<float>(1) += tag_points[i].y;
        center_tag.at<float>(2) += tag_points[i].z;
    }
    center_tag = center_tag / tag_points.size();

    //计算去质心坐标
    for(size_t i = 0; i < src_points.size(); ++i){
        src_points[i].x -= center_src.at<float>(0);
        src_points[i].y -= center_src.at<float>(1);
        src_points[i].z -= center_src.at<float>(2);
    }

    for(size_t i = 0; i < tag_points.size(); ++i){
        tag_points[i].x -= center_tag.at<float>(0);
        tag_points[i].y -= center_tag.at<float>(1);
        tag_points[i].z -= center_tag.at<float>(2);
    }
    //计算q*trans(q')
    cv::Mat A(3, 3, CV_32F, float(0));

    for(int i = 0; i < points_num; i++)
    {
        const float x1 = src_points[i].x;
        const float y1 = src_points[i].y;
        const float z1 = src_points[i].z;
        const float x2 = tag_points[i].x;
        const float y2 = tag_points[i].y;
        const float z2 = tag_points[i].z;

        A.at<float>(0,0) += x1*x2;
        A.at<float>(0,1) += x1*y2;
        A.at<float>(0,2) += x1*z2;
        A.at<float>(1,0) += y1*x2;
        A.at<float>(1,1) += y1*y2;
        A.at<float>(1,2) += y1*z2;
        A.at<float>(2,0) += z1*x2;
        A.at<float>(2,1) += z1*y2;
        A.at<float>(2,2) += z1*z2;
    }
//    std::cout << "finished A compute" << std::endl;
    //完成svd分解
    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//    std::cout << "svd finished!" << std::endl;
//    std::cout << u << ", " << vt << std::endl;
    cv::Mat R = u*vt;
//    std::cout << "R finished!" <<std::endl;
    cv::Mat t = center_src - R*center_tag;
//    std::cout << "compute t" << std::endl;
    Eigen::Matrix3f cur_rvec_matrix;
    cv::cv2eigen(R, cur_rvec_matrix);


    trans.block<3,3>(0,0) = cur_rvec_matrix;

    Eigen::Vector3f t_matrix;
    cv::cv2eigen(t, t_matrix);
    trans.block<3,1>(0,3) = t_matrix;

}
void pose::arucoInitial(const std::string fileName)
{
    cv::FileStorage fs(fileName, cv::FileStorage::READ);

    fs["cameraMatrix"] >> camera_matrics_;
    fs["distCoeffs"] >> camera_dist_coffs_;
    fs["captureNum"] >> total_capture_frame_;
    fs["distanceThresh"] >> distance_tresh_;
    fs["corners_x"] >> corners_x_;
    fs["corners_y"] >> corners_y_;
    fs.release();

    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    //marker length
    double length = 0.065;
    std::vector<int> ids_vec;
    //marker left up corner coordinate
    std::vector<double> corners_x = {0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48};
    std::vector<double> corners_y = {0, 0, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.08, 0.16, 0.24, 0.32, 0.40, 0.08, 0.16, 0.24, 0.32, 0.40};

    //CCW order is important
    //here we order the four corners for:left up, left bottom, right bottom and right up
    for(unsigned i = 0; i < corners_x.size(); ++i){
        std::vector<cv::Point3f> marker_corners(4, cv::Point3f(0, 0, 0));
        marker_corners[0].x = corners_x[i];
        marker_corners[0].y = corners_y[i];
        marker_corners[1].x = corners_x[i];
        marker_corners[1].y = corners_y[i] + length;
        marker_corners[2].x = corners_x[i] + length;
        marker_corners[2].y = corners_y[i] + length;
        marker_corners[3].x = corners_x[i] + length;
        marker_corners[3].y = corners_y[i];
        corners_.push_back(marker_corners);
        ids_vec.push_back(i + 1);
    }

    // create the board
    board_ptr_ = cv::aruco::Board::create(corners_, dictionary_, ids_vec);
}

void pose::markerDetect(const cv::Mat rgb_mat, const cv::Mat depth_mat, std::vector<cv::Point3f> &cornersVec)
{
    cv::Mat image_copy = rgb_mat.clone();
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    cv::aruco::detectMarkers(image_copy, dictionary_, corners, ids);

    if(ids.size() > 10){
        cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
    }
}

void pose::projection(cv::Mat depth_map, cv::Point2f uv, cv::Point3f &xyz)
{
    xyz.z = depth_map.at<ushort>(uv.x, uv.y);

}
}

