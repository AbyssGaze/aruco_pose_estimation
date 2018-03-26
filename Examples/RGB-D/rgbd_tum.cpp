#include <include/3DTransformation.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>

using namespace std;
using namespace cv;
using namespace pcl;
using namespace poseEstimate;

double FOCAL_X = 558.341390;
double FOCAL_Y = 558.387543;
double CX = 314.763671;
double CY = 240.992295;

#define READ_TRANS(TRANS, NAME) {Eigen::Matrix4f tmp_trans; ifstream(NAME.c_str(), std::ifstream::in | ifstream::binary).read((char*)tmp_trans.data(), 4 * 4 * sizeof(float)); (TRANS) = tmp_trans;}

int main (int argc, char** argv)
{

    // read the rgb-d data one by one
    ifstream index;
    string root_dir = argv[1];
    index.open ((root_dir).c_str());
    visualization::PCLVisualizer::Ptr vis(new visualization::PCLVisualizer);
    PointCloud<PointXYZRGB>::Ptr merge(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr merge_all(new PointCloud<PointXYZRGB>);

    vis->addCoordinateSystem(0.2);

    //从rgb图像中获取对应的特征点的3d和2d坐标
    while (!index.eof ())
    {
        string file;
        string name;
        index >> file;
        std::cout << file << endl;
        if (file.empty ())
          break;

        //    file = root_dir + "/" + file;

        Mat image;
        Mat depth_mat;
        //    cout << file << endl;
        // read data
        name = file + ".jpg";
        image = imread(name);
        name = file + ".png";
        depth_mat = imread(name, -1);
        if(!image.empty())
            imshow("rgb", image);
        waitKey(100);

        //产生mask区域
        string pose_file = file+".tra";
        Eigen::Matrix4f trans, trans_inv;
        READ_TRANS(trans, pose_file);
        trans(3,3) = 1;
        trans_inv = trans.inverse();
        //convert depth to cloud
        //
        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

        for (int y = 0; y < depth_mat.rows; ++y)
        {
            for (int x = 0; x < depth_mat.cols; ++x)
            {
                PointXYZRGB point_rgb;
                unsigned short d = depth_mat.at<unsigned short>(y, x);

                if (d != 0)
                {
                    point_rgb.z = (double)d / (double)1000; // Convert from mm to meters
                    point_rgb.x = (double)((x)-CX) * point_rgb.z / FOCAL_X;
                    point_rgb.y = (double)((y)-CY) * point_rgb.z / FOCAL_Y;

                    point_rgb.b = image.at<Vec3b>(y,x)[0];
                    point_rgb.g = image.at<Vec3b>(y,x)[1];
                    point_rgb.r = image.at<Vec3b>(y,x)[2];
                    cloud->points.push_back(point_rgb);
                }
            }
        }
        //cloud fileter
        transformPointCloud(*cloud, *cloud, trans_inv);
        *merge_all += *cloud;

        PassThrough<PointXYZRGB> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.009, 0.3);
        pass.filter (*cloud);

        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("x");
        pass.setFilterLimits (0.06, 0.5);
        pass.filter (*cloud);

        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (0.06, 0.5);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud);
        //mask and erosion

        *merge += *cloud;
        vis->removeAllPointClouds();
        vis->addPointCloud(merge);
        vis->spin();

    }
    vis->removeAllPointClouds();
    vis->addPointCloud(merge_all);
    vis->spin();
    vis->removeAllPointClouds();
    vis->addPointCloud(merge);
    vis->spin();
    //对merge的点云进行处理
    cout<<"statistical outlier removal begins ..."<<endl;
    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud (merge);
    sor.setMeanK (20);
    sor.setStddevMulThresh (1.4);
    sor.filter (*merge);
    vis->removeAllPointClouds();
    vis->addPointCloud(merge);
    vis->spin();
    //
    cout<<"moving least square begins ..."<<endl;
    PointCloud<PointXYZRGB>::Ptr cloud_mls(new PointCloud<PointXYZRGB>);

    search::KdTree<PointXYZRGB>::Ptr tree (new pcl::search::KdTree<PointXYZRGB>);
    // Output has the PointNormal type in order to store the normals calculated by MLS
    MovingLeastSquares<PointXYZRGB, PointXYZRGB> mls;
    mls.setComputeNormals (true);
    mls.setInputCloud (merge);
    mls.setPolynomialFit (true);
    mls.setSearchMethod (tree);
    mls.setSearchRadius (0.01);
    mls.process (*cloud_mls);
    vis->removeAllPointClouds();
    vis->addPointCloud(cloud_mls);
    vis->spin();
    cout<<"Down sampling the point cloud..."<<endl;
    VoxelGrid<PointXYZRGB> vf;
    vf.setInputCloud (cloud_mls);
    vf.setLeafSize (0.001, 0.001, 0.001);
    vf.filter (*cloud_mls);

    //完成mesh的构建
    NormalEstimationOMP<PointXYZRGB, Normal> ne;
    search::KdTree<PointXYZRGB>::Ptr tree1 (new search::KdTree<PointXYZRGB>);
    tree1->setInputCloud (cloud_mls);
    ne.setInputCloud (cloud_mls);
    ne.setSearchMethod (tree1);
    ne.setKSearch (20);
    PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
    ne.compute (*normals);

    // Concatenate the XYZ and normal fields*
    PointCloud<PointXYZRGBNormal>::Ptr cloud_with_normals (new PointCloud<PointXYZRGBNormal>);
    concatenateFields(*cloud_mls, *normals, *cloud_with_normals);

    // Create search tree*
    search::KdTree<PointXYZRGBNormal>::Ptr tree2 (new search::KdTree<PointXYZRGBNormal>);
    tree2->setInputCloud (cloud_with_normals);

    cout << "begin marching cubes reconstruction" << endl;

//    MarchingCubesRBF<PointXYZRGBNormal> mc;
//    PolygonMesh::Ptr triangles(new PolygonMesh);
//    mc.setInputCloud (cloud_with_normals);
//    mc.setSearchMethod (tree2);
//    mc.reconstruct (*triangles);
    GreedyProjectionTriangulation<PointXYZRGBNormal> gp3;   //定义三角化对象
    pcl::PolygonMesh triangles;                //存储最终三角化的网络模型

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.05);  //设置连接点之间的最大距离，（即是三角形最大边长）

    // 设置各参数值
    gp3.setMu (2.5);  //设置被样本点搜索其近邻点的最远距离为2.5，为了使用点云密度的变化
    gp3.setMaximumNearestNeighbors (100);    //设置样本点可搜索的邻域个数
    gp3.setMaximumSurfaceAngle(M_PI/4); // 设置某点法线方向偏离样本点法线的最大角度45
    gp3.setMinimumAngle(M_PI/18); // 设置三角化后得到的三角形内角的最小的角度为10
    gp3.setMaximumAngle(2*M_PI/3); // 设置三角化后得到的三角形内角的最大角度为120
    gp3.setNormalConsistency(false);  //设置该参数保证法线朝向一致

    // Get result
    gp3.setInputCloud (cloud_with_normals);     //设置输入点云为有向点云
    gp3.setSearchMethod (tree2);   //设置搜索方式
    gp3.reconstruct (triangles);  //重建提取三角化

    vis->removeAllPointClouds();
    vis->addPolygonMesh(triangles);
    vis->spin();

//    cout << "begin marching Poisson reconstruction" << endl;

//    Poisson<PointXYZRGBNormal> pn ;
//    pn.setSearchMethod(tree2) ;
//    pn.setInputCloud(cloud_with_normals) ;
//    pn.setConfidence(false) ;//设置置信标志，为true时，使用法线向量长度作为置信度信息，false则需要对法线进行归一化处理
//    pn.setManifold(false) ;//设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
//    pn.setOutputPolygons(false) ;//设置是否输出为多边形
//    pn.setIsoDivide(8) ;
//    pn.setSamplesPerNode(3) ;//设置每个八叉树节点上最少采样点数目

//    pn.performReconstruction(triangles) ;
//    vis->removeAllPointClouds();
//    vis->addPolygonMesh(triangles);
//    vis->spin();
}
