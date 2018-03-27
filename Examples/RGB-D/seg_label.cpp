#include <pcl/visualization/cloud_viewer.h>

#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/registration/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <opencv2/opencv.hpp>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/bilateral.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/flann_search.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace pcl;
using namespace cv;

//! read frame pose from file
#define READ_TRANS(TRANS, NAME) {Eigen::Matrix4f tmp_trans; ifstream(NAME.c_str(), std::ifstream::in | ifstream::binary).read((char*)tmp_trans.data(), 4 * 4 * sizeof(float)); (TRANS) = tmp_trans;}

typedef PointXYZRGBL PointT;
typedef visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

//! camera intrinsic
double fx = 558.341390;
double fy = 558.387543;
double cx = 314.763671;
double cy = 240.992295;
set<int> label_set;

//! Convert point cloud to depth mat
Mat convert2Map(PointCloud<PointT>::Ptr cloud){
    Mat resultMat = Mat::zeros(480, 640, CV_8UC1);
    for(unsigned i = 0; i < cloud->points.size(); ++i){
        PointT point;
        point = cloud->points[i];
        int u, v;
        u = point.x * fx / point.z + cx;
        v = point.y * fy / point.z + cy;
        if (v < 480 && v > 0 && u < 640 && u > 0)
            resultMat.at<uchar>(v, u) = 255;
    }
    return resultMat;
}

//! Convert point cloud to depth mat
Mat convert2Map(PointCloud<PointXYZL>::Ptr cloud){
    Mat resultMat = Mat::zeros(480, 640, CV_8UC1);
    for(unsigned i = 0; i < cloud->points.size(); ++i){
        PointXYZL point;
        point = cloud->points[i];
        int u, v;
        u = point.x * fx / point.z + cx;
        v = point.y * fy / point.z + cy;
        if (v < 480 && v > 0 && u < 640 && u > 0)
            resultMat.at<uchar>(v, u) = 255;
    }
    return resultMat;
}

//! Convert PointXYZL cloud to vector cloud, different vectord coresponding to different label
void convertLabledCloud2Vec(PointCloud<PointT>::Ptr cloud, vector<PointCloud<PointT>::Ptr>& cloud_vec){
    for(unsigned i = 0; i < cloud->points.size(); ++i){
        int label = cloud->points[i].label;
        if(label >= (int)cloud_vec.size()){
            for(int j = 0; j <= label; ++j){
                PointCloud<PointT>::Ptr cloud_copy(new PointCloud<PointT>);
                cloud_vec.push_back(cloud_copy);
            }
            cloud_vec[label]->points.push_back(cloud->points[i]);
        }
        else{
            cloud_vec[label]->points.push_back(cloud->points[i]);
        }
    }
}

//! Convert PointXYZL cloud to vector cloud, different vectord coresponding to different label
void convertLabledCloud2Vec(PointCloud<PointXYZL>::Ptr cloud, vector<PointCloud<PointXYZL>::Ptr>& cloud_vec){

    for(unsigned i = 0; i < cloud->points.size(); ++i){
        int label = cloud->points[i].label;
        if(label >= (int)cloud_vec.size()){
            for(int j = 0; j <= label; ++j){
                PointCloud<PointXYZL>::Ptr cloud_copy(new PointCloud<PointXYZL>);
                cloud_vec.push_back(cloud_copy);
            }
            cloud_vec[label]->points.push_back(cloud->points[i]);
        }
        else{
            cloud_vec[label]->points.push_back(cloud->points[i]);
        }
    }
}

/**\brief Compute overlap ratio between src mat and tag vector, and return the max ratio index.
 * \param[in] src
 * \param[in] tag a list mask vector
 * \return max ratio index
 * */
int computeOverlap(Mat src, vector<Mat> tag){
    double ratio = 0;
    int index = 0;
    for(unsigned i = 0; i < tag.size(); ++i){
        Mat res;
        bitwise_and(src, tag[i], res);
        double and_ratio = sum(res)[0];
        if(and_ratio > ratio){
            ratio = and_ratio;
            index = i;
        }
    }
    return index;
}

void labelPointCloud(PointCloud<PointT>::Ptr& cloud_src, PointCloud<PointXYZL>::Ptr& cloud_tag){
    Mat resultMat = Mat::zeros(480, 640, CV_8UC1);
    for(unsigned i = 0; i < cloud_tag->points.size(); ++i){
        PointXYZL point;
        point = cloud_tag->points[i];
        int u, v;
        u = point.x * fx / point.z + cx;
        v = point.y * fy / point.z + cy;
        if (v < 480 && v > 0 && u < 640 && u > 0)
            resultMat.at<uchar>(v, u) = point.label;
    }
    for(unsigned i = 0; i < cloud_src->points.size(); ++i){
        PointT point;
        point = cloud_src->points[i];
        int u, v;
        u = point.x * fx / point.z + cx;
        v = point.y * fy / point.z + cy;
        if (v < 480 && v > 0 && u < 640 && u > 0)
            cloud_src->points[i].label = resultMat.at<uchar>(v, u);
    }
}

static bool cmp(pair<int, int> a, pair<int, int> b){
    return a.second > b.second;
}

/**\brief Supervoxel segmentation.
 * \param[in] cloud you need to segment
 * \param[out] seg_cloud supervoxel segmentation result
 * */
void superVoxleSeg(PointCloud<PointT>::Ptr &cloud, PointCloud<PointXYZL>::Ptr &seg_cloud){
    float voxel_resolution = 0.005f;

    float seed_resolution = 0.04f;

    float color_importance = 0.8f;

    float spatial_importance = 0.4f;

    float normal_importance = 1.5f;

    SupervoxelClustering<PointXYZRGB> super (voxel_resolution, seed_resolution);
//   super.setUseSingleCameraTransform (false);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);

    map <uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_clusters;

    PointCloud<PointXYZRGB>::Ptr cloud_copy(new PointCloud<PointXYZRGB>);
    copyPointCloud(*cloud, *cloud_copy);
    ///supervoxel segmentation///
    console::print_highlight ("Extracting supervoxels!\n");
    super.setInputCloud (cloud_copy);
    super.extract (supervoxel_clusters);
    seg_cloud = super.getLabeledCloud();
}

//! Cloud filter
void cloudPassFilter(PointCloud<PointT>::Ptr& cloud){

    PassThrough<PointT> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.009, 0.3);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.05, 0.5);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (0.05, 0.5);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud);

//    StatisticalOutlierRemoval<PointT> sor;
//    sor.setInputCloud (cloud);
//    sor.setMeanK (10);
//    sor.setStddevMulThresh (0.5);
//    sor.filter (*cloud);
}

//! Convert cloud to different type dataset structure
void cloudRenderMat(PointCloud<PointT>::Ptr cloud, vector<vector<pair<double, int>>> &cloud_vec){
    cloud_vec.resize(307200);
    for(unsigned i = 0; i < cloud->points.size(); ++i){
        int u, v;
        PointT point;
        point = cloud->points[i];
        u = point.x * fx /point.z + cx;
        v = point.y * fy /point.z + cy;
        if(v < 480 && v > 0 && u < 640 && u > 0){
            cloud_vec[v*640 + u].push_back({point.z * (double)1000, point.label});

        }
        label_set.insert(point.label);
    }
}

//! Convert cloud to different type dataset structure
void cloudRenderMatch(PointCloud<PointT>::Ptr &cloud, vector<vector<pair<double, int>>> cloud_vec){
//    //compute normal
//    PointCloud <Normal>::Ptr normals(new pcl::PointCloud <Normal>);
//    search::Search<PointT>::Ptr tree = boost::shared_ptr<search::Search<PointT> >(new search::KdTree<PointT>);

//    NormalEstimation<PointT, Normal> normal_estimator;
//    normal_estimator.setSearchMethod(tree);
//    normal_estimator.setInputCloud(cloud);
//    normal_estimator.setKSearch(100);
//    normal_estimator.compute(*normals);

    for(unsigned i = 0; i < cloud->points.size(); ++i){
        int u, v;
        double res = 18;
        PointT point;
        point = cloud->points[i];
        u = point.x * fx /point.z + cx;
        v = point.y * fy /point.z + cy;
        if(v < 480 && v > 0 && u < 640 && u > 0){
            for(unsigned k = 0; k < cloud_vec[v*640 + u].size(); ++k){
                pair<double,int> z = cloud_vec[v*640 + u][k];
                if(abs(z.first - point.z*1000) < res){
                    res = abs(z.first - point.z*1000);
                    cloud->points[i].label = z.second + 1;
                }
            }
        }
    }
}

/**\brief Supervoxel segmentation.
 * \param[in] v1 one normal
 * \param[in] v2 another normal
 * \param[in] in_degree result is degree or angle model
 * \return normals angle
 * */
double getAngle3D(Eigen::Vector3f &v1, Eigen::Vector3f &v2, const bool in_degree)
{
    // Compute the actual angle
    double rad = v1.normalized().dot(v2.normalized());
    if (rad < -1.0)
        rad = -1.0;
    else if (rad >  1.0)
        rad = 1.0;
//    std::cout << "[" << v1 << "], ["<< v2 << "], " << acos(rad) << std::endl;
    return (in_degree ? acos(rad) * 180.0 / M_PI : acos(rad));
}


int main(int argc, char** argv){
    if (argc < 2)
    {
        PCL_INFO ("This is a utility program meant to render a mesh from a TSDF Volume, which can be saved to disk via TSDFVolumeOctree::save(const std::string &filename).\n");
        PCL_INFO ("Usage: %s ../input_label.txt", argv[0]);
        return (1);
    }
    ifstream infoFile(argv[1], ifstream::in);
    int cnt_id = 0;
    infoFile >> cnt_id;
    string cloudFileStr, poseFileStr, labelCloudStr, labelPoseStr;
    infoFile >> cloudFileStr >> poseFileStr >> labelCloudStr >> labelPoseStr;
    //read file
    ifstream cloudFile(cloudFileStr, ifstream::in);
    ifstream poseFile(poseFileStr, ifstream::in);
    ifstream labelCloud(labelCloudStr, ifstream::in);
    ifstream labelPose(labelPoseStr, ifstream::in);
    string labelCloudFileName;
    string labelPoseFileName;
    string cloudFileName;
    string poseFileName;
    stringstream pcd_file;
    Eigen::Matrix4f trans;

    PointCloud<PointT>::Ptr cloud(new PointCloud<PointT>);
    PointCloud<PointT>::Ptr merged(new PointCloud<PointT>);
    PointCloud<PointT>::Ptr model(new PointCloud<PointT>);

    PointCloud<PointXYZL>::Ptr cloud_copy(new PointCloud<PointXYZL>);


    visualization::PCLVisualizer::Ptr vis(new visualization::PCLVisualizer);
    vis->addCoordinateSystem(0.1);
    //将已完成标记的点云读入，并将其合并到Merged，构建好全局model
    int model_id = 0;
    while(cloudFile >> cloudFileName && poseFile >> poseFileName)
    {
        READ_TRANS(trans, poseFileName);

        if(io::loadPCDFile<PointT>(cloudFileName, *cloud) == -1){
            cout << "The cloud name is wrong!" << endl;
            break;
        }
        if(model_id++ < 10)
        {
            for(unsigned i = 0; i < cloud->points.size(); ++i)
            {
                cloud->points[i].label = cloud->points[i].label + 4;
            }
        }
//        trans(3,3) = 1;
//        Eigen::Matrix4f trans_inv = trans.inverse();
        transformPointCloud (*cloud, *cloud, trans);
        cloudPassFilter(cloud);

        *model += *cloud;
//        vis->removeAllPointClouds();
//        vis->addPointCloud<PointT>(model, "model");
//        vis->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

//        vis->spin ();
    }


    while(labelCloud >> labelCloudFileName && labelPose >> labelPoseFileName){
        //读取待标记的点云
        if(io::loadPCDFile<PointT>(labelCloudFileName, *cloud) == -1){
            cout << "The cloud name is wrong!" << endl;
        }

        READ_TRANS(trans, labelPoseFileName);
        Eigen::Matrix4f trans_inverse = trans.inverse();

        transformPointCloud (*cloud, *cloud, trans);

        cloudPassFilter(cloud);

        //convert cloud view point from artag coordinate to camera coordinate
        transformPointCloud (*cloud, *cloud, trans_inverse);
        transformPointCloud (*model, *model, trans_inverse);

        vis->removeAllPointClouds();
        vis->addPointCloud (cloud, ColorHandlerT (cloud, 0.0, 255.0, 0.0), "scene");
        vis->addPointCloud (model, ColorHandlerT (model, 0.0, 0.0, 255.0), "object_aligned");
        vis->spin ();
        //将model渲染到该视角下的深度图，并且vector的size为640*480，存储的为该像素下多个z和对应的label
        vector<vector<pair<double, int>>> model_render;
        cloudRenderMat(model, model_render);
        cout << "**********model size: " << label_set.size() << "****************" << endl;
        for (std::set<int>::iterator it=label_set.begin(); it!=label_set.end(); ++it)
            std::cout << ' ' << *it;
        cout << endl;
        //给点云初始化对应的label信息，但是由于位姿估计存在一定的偏差，所以可能会有一些匹配点没有渲染到，所以通过分割来完成统计
        cloudRenderMatch(cloud, model_render);

        copyPointCloud(*cloud, *cloud_copy);

        vis->removeAllPointClouds();
        vis->addPointCloud (cloud_copy, "labeled voxels");
        vis->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

        vis->spin ();

        //超体像素分割后，统计每个voxel中的点的label来完成标记
        PointCloud<PointXYZL>::Ptr cloud_seg(new PointCloud<PointXYZL>);
        superVoxleSeg(cloud, cloud_seg);

        vis->removeAllPointClouds();
        vis->addPointCloud (cloud_seg, "labeled voxels");
        vis->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

        vis->spin ();

        vector<PointCloud<PointT>::Ptr> cloud_label_vec;
        convertLabledCloud2Vec(cloud, cloud_label_vec);

        vector<PointCloud<PointXYZL>::Ptr> cloud_seg_vec;
        convertLabledCloud2Vec(cloud_seg, cloud_seg_vec);


        vector<Mat> cloudMat_vec;
        for(unsigned i = 0; i < cloud_label_vec.size(); ++i){
            Mat mat = convert2Map(cloud_label_vec[i]);
            cloudMat_vec.push_back(mat);
        }
        PointCloud<PointXYZL>::Ptr labelCloud(new PointCloud<PointXYZL>);

        for(unsigned i = 0; i < cloud_seg_vec.size(); ++i){
            Mat segMat = convert2Map(cloud_seg_vec[i]);
            int index = computeOverlap(segMat, cloudMat_vec);
            for(unsigned k = 0; k < cloud_seg_vec[i]->points.size(); ++k){
                PointXYZL point = cloud_seg_vec[i]->points[k];
                point.label = index;
                labelCloud->points.push_back(point);
            }
        }
//        trans_inverse = trans_inverse.inverse().eval();
//        transformPointCloud (*model, *model, trans_inverse);

        transformPointCloud (*model, *model, trans);

        if(labelCloud->points.size() > 0){
            vis->removeAllPointClouds();
            vis->addPointCloud (labelCloud, "labeled voxels");
            vis->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

            vis->spin ();

            labelPointCloud(cloud, labelCloud);
            //save cloud
            console::print_highlight ("If you want save, press s, else press q!\n");

            char key;
            cin >> key;
            if(key == 's'){
                pcd_file.str("");
                pcd_file << "/home/cy/Projects/dataset/labeled_0628/xtion_pro_live_0628_" << setw( 5 ) << setfill( '0' ) << cnt_id << ".pcd";
                io::savePCDFile<PointT> (pcd_file.str(), *cloud, true);
                pcd_file.str("");
                pcd_file << "/home/cy/Projects/dataset/labeled_0628/xtion_pro_live_0628_" << setw( 5 ) << setfill( '0' ) << cnt_id << ".tra";
                fstream fs(pcd_file.str(), ios::out | ios::binary);
                fs.write((char*)trans.data(), 4 * 4 * sizeof(float));
                cnt_id++;
            }

        }

    }

}
