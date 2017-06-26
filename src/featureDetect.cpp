#include <include/featureDetect.h>

namespace poseEstimate {

void feature::surfFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor)
{
    int minHessian = 400;

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );

    detector->detect( rgb_image, keyPoints);

    detector->compute(rgb_image, keyPoints, descriptor);
}

void feature::siftFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor)
{
    int minHessian = 400;

    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create( minHessian );

    detector->detect( rgb_image, keyPoints);

    detector->compute(rgb_image, keyPoints, descriptor);
}

//void feature::orbFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor)
//{

//    cv::ORB detector(1000,1.1f,8,31,0,2,cv::ORB::HARRIS_SCORE,31);

//    detector.detect( rgb_image, keyPoints);

//    detector.compute(rgb_image, keyPoints, descriptor);
//}

//void feature::fastFeatureDetect(cv::Mat rgb_image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor)
//{
//    int minHessian = 400;

//    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );

//    detector->detect( rgb_image, keyPoints);

//    detector->compute(rgb_image, keyPoints, descriptor);
//}
void feature::readParams(std::string param)
{
    cv::FileStorage fs(param, cv::FileStorage::READ);

    fs["cameraMatrix"] >> camera_matrics_;
    fs["distCoeffs"] >> camera_dist_coffs_;
    fs["captureNum"] >> total_capture_frame_;
    fs["distanceThresh"] >> distance_tresh_;
    fs["arucoLength"] >> length_;
    fs["arucoCornersX"] >> corners_x_;
    fs["arucoCornersY"] >> corners_y_;
    std::cout << length_ << std::endl;
    fs.release();
}
void feature::createBoard()
{
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    //marker length
    double length = length_;
    //marker left up corner coordinate
    std::vector<double> corners_x(corners_x_.rows, 0);
    //= {0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48};
    std::vector<double> corners_y(corners_y_.rows, 0);
    //= {0, 0, 0, 0, 0, 0, 0, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.08, 0.16, 0.24, 0.32, 0.40, 0.08, 0.16, 0.24, 0.32, 0.40};
//    std::cout << "rows:" << corners_x_.rows << std::endl;
    for(int i = 0; i < corners_x_.rows; ++i){
        corners_x[i] = (corners_x_.at<double>(i, 0));
//        std::cout << corners_x[i] << " ";
    }
//    std::cout << std::endl;
    for(int i = 0; i < corners_y_.rows; ++i){
        corners_y[i] = (corners_y_.at<double>(i, 0));
//        std::cout << corners_y[i] << " ";
    }
//    std::cout << std::endl;
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
        marker_corner_vec_.push_back(marker_corners);
        ids_vec_.push_back(i + 1);
    }

    // create the board
    board_ptr_ = cv::aruco::Board::create(marker_corner_vec_, dictionary_, ids_vec_);

    //translate the board to mat
//    aruco::drawPlanarBoard(board_ptr_, Size(3000, 3000), board_image_, 70, 1);
}

bool feature::arucoDetect(cv::Mat rgb_image, cv::Mat depth_image, std::vector<int> ids3d, std::vector<std::vector<cv::Point3f> > &corners3d, Eigen::Matrix4f& trans)
{
    cv::Mat image_copy = rgb_image.clone();
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    cv::aruco::detectMarkers(image_copy, dictionary_, corners, ids);
//    for(unsigned i = 0; i < corners.size(); ++i){
//        std::cout << "-------------ids: " << ids[i] << "--------------------" << std::endl;
//        for(unsigned j = 0; j < corners[i].size(); ++j){
//            std::cout << corners[i][j].x << ", " << corners[i][j].y << std::endl;
//        }
//    }
//    std::cout << "camera matrix:" << camera_matrics_ << std::endl;
    //convert 2d corners to 3d corners
    bool result = false;
//    std::cout << "----size of ids: " << ids.size() << std::endl;
    if(ids.size() > 3)
    {
        result = true;
        for(unsigned i = 0; i < corners.size(); ++i)
        {
            std::vector<cv::Point3f> corner;
            for(unsigned j = 0; j < corners[i].size(); ++j)
            {
                int u, v;
                u = corners[i][j].x;
                v = corners[i][j].y;
                cv::Point3f point;
                unsigned short d = depth_image.at<unsigned short>(v, u);

                if (d != 0)
                {
                    point.z = (double)d / (double)1000; // Convert from mm to meters
                    point.x = ((double)u - camera_matrics_.at<double>(0,2)) * point.z / camera_matrics_.at<double>(0,0);
                    point.y = ((double)v - camera_matrics_.at<double>(1,2)) * point.z / camera_matrics_.at<double>(1,1);
                    corner.push_back(point);
                }

            }
            if(corner.size() == 4){
                corners3d.push_back(corner);
                ids3d.push_back(ids[i]);
            }
        }
    //    std::cout << "marker size: \n" << marker_corner_vec_.size() << std::endl;
        std::vector<cv::Point3f> board_corners, detected_corners;
        for(unsigned i = 0; i < ids3d.size(); ++i)
        {
            for(int j = 0; j < 4; ++j){
    //            std::cout << "id: " << ids3d[i] << std::endl;
                board_corners.push_back(marker_corner_vec_[ids3d[i] - 1][j]);
                detected_corners.push_back(corners3d[i][j]);
            }
        }
    //    Eigen::Matrix4f trans;
    //    Eigen::Matrix3f trans_r;
    //    std::cout << "start :" << board_corners.size() << ", " << detected_corners.size() << std::endl;
        poseEstimation(detected_corners, board_corners, trans);
    //    std::cout << "trans:\n" << trans << std::endl;
    //    std::cout << "trans:" << trans(0,0) << std::endl;
        //对trans进行反罗德里格斯变换
        cv::Mat r_mat(3,3,CV_32F, float(0));
    //    std::cout << "r_mat:" << r_mat << std::endl;
    //    std::cout << "r_mat:" << r_mat.at<float>(0, 0) << std::endl;

        r_mat.at<float>(0, 0) = trans(0,0);
        r_mat.at<float>(0, 1) = trans(0,1);
        r_mat.at<float>(0, 2) = trans(0,2);
        r_mat.at<float>(1, 0) = trans(1,0);
        r_mat.at<float>(1, 1) = trans(1,1);
        r_mat.at<float>(1, 2) = trans(1,2);
        r_mat.at<float>(2, 0) = trans(2,0);
        r_mat.at<float>(2, 1) = trans(2,1);
        r_mat.at<float>(2, 2) = trans(2,2);
    //    std::cout << "finished r_mat:" << std::endl;

    //    trans_r = trans.block<3,3>(0,0);
    //    cv::eigen2cv(trans_r, r_mat);
        cv::Vec3d t;
        t[0] = trans(0,3);
        t[1] = trans(1,3);
        t[2] = trans(2,3);
        cv::Mat R;
    //    std::cout << "start rodrigues:" << r_mat << std::endl;
        cv::Rodrigues(r_mat, R);
    //    std::cout << "finished rodrigues:" << std::endl;


        //    if(ids.size() > 10){
            cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
            cv::aruco::drawAxis(image_copy, camera_matrics_, camera_dist_coffs_, R, t, 0.1);

    //        int valid = cv::aruco::estimatePoseBoard(corners, ids, board_ptr_, camera_matrics_, camera_dist_coffs_, cur_rvec_, cur_tvec_);
    //        if(valid > 0){
    //            cv::aruco::drawAxis(image_copy, camera_matrics_, camera_dist_coffs_, cur_rvec_, cur_tvec_, 0.1);
    //        }
            cv::imshow("rgb_copy", image_copy);

    //    }
    }
//    else
//    {
//        std::cout << "----size of ids: ";
//        for(unsigned j = 0; j < ids.size(); ++j)
//            std::cout << ids[j] << " ";
//        std::cout << std::endl;

//    }
    return result;

}

void feature::featureMatch(cv::Mat &desc_1, cv::Mat &desc_2, std::vector<cv::DMatch> &matches)
{
    cv::BFMatcher matcher;
    matcher.match(desc_1, desc_2, matches);

    //RANSAC algorithm
}


}
