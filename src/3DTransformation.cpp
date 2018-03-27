#include <include/3DTransformation.h>

namespace poseEstimate
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        trans(3,3) = 1;
    }

}

