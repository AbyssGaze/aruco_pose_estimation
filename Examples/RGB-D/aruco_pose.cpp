#include <include/featureDetect.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <pcl/io/openni2_grabber.h>
#include <pcl/io/png_io.h>
#include <fstream>

#include <Eigen/Geometry>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace pcl;


poseEstimate::feature fb;

int num = 0;

Eigen::Matrix4f trans, old_trans;

Eigen::Vector3f euler_angles, euler_angles_old, t_old, t;

//! Current rotation matrix and old rotation matrix
Eigen::Matrix3f rotation_matrix, rotation_matrix_old;

//! Input: depth file path with name...
stringstream depth_file, rgb_file, pose_file;

//! To save depth png file
vector<int> params;

//! Callback function get depth and rgb frame and compute artag pose
void callback (const boost::shared_ptr<io::Image>& rgb, const boost::shared_ptr<io::DepthImage>& depth, float constant)
{
    int sizes[2] = {480, 640};
    Mat rgb_mat(2, sizes, CV_8UC3);
    Mat depth_mat(2, sizes,CV_16UC1);

    rgb->fillRGB(640, 480, rgb_mat.data, 640*3);
    cvtColor(rgb_mat,rgb_mat,CV_RGB2BGR);

    depth->fillDepthImageRaw(640, 480, (unsigned short*) depth_mat.data);

    vector<vector<Point3f>> corners;
    vector<int> ids;
    bool valid = fb.arucoDetect(rgb_mat, depth_mat, ids, corners, trans);
    if(valid){
        if(num == 0)//initial frame
        {
            old_trans = trans;
            num++;
            rotation_matrix_old = trans.block<3,3>(0,0);
            t_old = trans.block<3,1>(0,3);
            euler_angles_old = rotation_matrix_old.eulerAngles(2,1,0);
            cout << "---------------------INITIAL-------------------" << endl;
            cout << "rotation:    " << euler_angles_old.transpose() << endl;
            cout << "translation: " << t_old.transpose() << endl;

        }
        else
        {
            //save depth and rgb mat when y axis translation than a threshold
            rotation_matrix = trans.block<3,3>(0,0);
            t = trans.block<3,1>(0,3);
            euler_angles = rotation_matrix.eulerAngles(2,1,0);//zyx sequence，roll pitch yaw sequence

            if(abs(euler_angles[1] - euler_angles_old[1])>0.3 || (t - t_old).norm() > 0.2)//0.3 radian and 0.2 meters
            {
                cout << "---------------------EULER-------------------" << endl;
                cout << "error:   " << abs(euler_angles[1] - euler_angles_old[1]) << endl;
                cout << "current: " << euler_angles.transpose() << endl;
                cout << "old:     " << euler_angles_old.transpose() << endl;
                cout << "---------------------TRANS-------------------" << endl;
                cout << "current: " << t_old.transpose() << endl;
                cout << "old:     " << t.transpose() << endl;

                t_old = t;
                euler_angles_old = euler_angles;

                //save pose, depth mat and rgb mat
                rgb_file.str("");
                depth_file.str("");
                pose_file.str("");

                rgb_file << "/home/cy/Projects/dataset/xtion_pro_live/xtion_pro_live_box_1_" << setw( 5 ) << setfill( '0' ) << num << ".jpg";
                depth_file << "/home/cy/Projects/dataset/xtion_pro_live/xtion_pro_live_box_1_" << setw( 5 ) << setfill( '0' )<< num << ".png";
                pose_file << "/home/cy/Projects/dataset/xtion_pro_live/xtion_pro_live_box_1_" << setw( 5 ) << setfill( '0' ) << num << ".tra";

                imwrite(rgb_file.str(), rgb_mat);
                imwrite(depth_file.str(), depth_mat, params);

                fstream fs(pose_file.str(), ios::out | ios::binary);
                fs.write((char*)trans.data(), 4 * 4 * sizeof(float));
                num++;
            }
        }
    }

}

int main (int argc, char** argv)
{
    if(argc < 2){
        cout << "The input arguments less than neccesary!" << endl;
        cout << "aruco_pose ../param.ymal" << endl;
        return -1;
    }
    ///read parameters
    fb.readParams(argv[1]);
    fb.createBoard();

    params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    params.push_back(0);

    Grabber* interface = new io::OpenNI2Grabber("#1");
    if (interface == 0)
        return -1;
    boost::function<void (const boost::shared_ptr<io::Image>&, const boost::shared_ptr<io::DepthImage>&, float)> f(&callback);
    interface->registerCallback (f);

    interface->start ();
    while (true)
    {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
        if(waitKey(30) == 'q')
            break;
    }

    interface->stop ();
    return 0;
}
