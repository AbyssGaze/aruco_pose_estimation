

#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

using namespace std;
using namespace pcl;
using namespace cv;


#define READ_TRANS(TRANS, NAME) {Eigen::Matrix4f tmp_trans; ifstream(NAME.c_str(), std::ifstream::in | ifstream::binary).read((char*)tmp_trans.data(), 4 * 4 * sizeof(float)); (TRANS) = tmp_trans;}

//! camera instrincs
double FOCAL_X = 558.341390;
double FOCAL_Y = 558.387543;
double CX = 314.763671;
double CY = 240.992295;

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cout << "you need put :" << endl;
        cout << "./pcl_view ../input.txt" << endl;
        return 1;
    }
    // read the rgb-d data one by one
    ifstream index;
    int num = 0;

//    if(pcl::console::parse(argc, argv, "-u", num))
//        cout << "the start num is:" << num << endl;


    string root_dir = argv[1];

    index.open ((root_dir).c_str());
    visualization::PCLVisualizer::Ptr vis(new visualization::PCLVisualizer);
    stringstream pcd_file, pose_file;

    vis->addCoordinateSystem(0.2);
    //get features 3d coordinate and 2d coordinate from rgb image
    index >> num;
    while (!index.eof ())
    {
        PointCloud<PointXYZRGBL>::Ptr cloud(new PointCloud<PointXYZRGBL>);

        string file;
        string name;
        index >> file;
        if (file.empty ())
          break;

        Mat rgb_mat;
        Mat depth_mat;
        //    cout << file << endl;
        // read data
        name = file + ".jpg";
        rgb_mat = imread(name);
        name = file + ".png";
        depth_mat = imread(name, -1);

        //generate mask regions
        string pose_name = file+".tra";
        Eigen::Matrix4f trans, trans_inv;
        READ_TRANS(trans, pose_name);
        trans(3,3) = 1;
        trans_inv = trans.inverse();

        for (int y = 0; y < depth_mat.rows; ++y)
        {
            for (int x = 0; x < depth_mat.cols; ++x)
            {
                PointXYZRGBL point_rgb;
                unsigned short d = depth_mat.at<unsigned short>(y, x);

                if (d != 0)
                {
                    point_rgb.z = (double)d / (double)1000; // Convert from mm to meters
                    point_rgb.x = (double)((x)-CX) * point_rgb.z / FOCAL_X;
                    point_rgb.y = (double)((y)-CY) * point_rgb.z / FOCAL_Y;

                    point_rgb.b = rgb_mat.at<Vec3b>(y,x)[0];
                    point_rgb.g = rgb_mat.at<Vec3b>(y,x)[1];
                    point_rgb.r = rgb_mat.at<Vec3b>(y,x)[2];
                    point_rgb.label = 0;
                    cloud->points.push_back(point_rgb);
                }
            }
        }
        //save cloud and pose
        pcd_file.str("");
        pose_file.str("");

        pcd_file << "/home/cy/Projects/dataset/xtion_pro_0628/pcd_pose/xtion_pro_live_0628_" << setw( 5 ) << setfill( '0' ) << num << ".pcd";
        pose_file << "/home/cy/Projects/dataset/xtion_pro_0628/pcd_pose/xtion_pro_live_0628_" << setw( 5 ) << setfill( '0' ) << num << ".tra";
        num ++;
        io::savePCDFile<PointXYZRGBL> (pcd_file.str(), *cloud, true);

        fstream fs(pose_file.str(), ios::out | ios::binary);
        fs.write((char*)trans_inv.data(), 4 * 4 * sizeof(float));
    }



}
