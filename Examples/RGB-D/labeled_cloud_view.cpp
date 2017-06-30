#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

#include <vector>

using namespace std;
using namespace pcl;

int main(int argc, char**argv)
{
    if(argc < 2)
    {
        cout << "you need put :" << endl;
        cout << "./labeled_cloud_view ../input.txt" << endl;
        return 1;
    }
    // read the rgb-d data one by one
    ifstream infoFile(argv[1], ifstream::in);
    visualization::PCLVisualizer::Ptr vis(new visualization::PCLVisualizer);
    PointCloud<PointXYZL>::Ptr cloud(new PointCloud<PointXYZL>);

    string cloud_name;

    while(infoFile >> cloud_name)
    {
        if(io::loadPCDFile<PointXYZL>(cloud_name, *cloud) == -1){
            cout << "The cloud name is wrong!" << endl;
            break;
        }
        vis->removeAllPointClouds();
        vis->addPointCloud (cloud, "labeled voxels");
        vis->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");
        vis->spin ();
    }


}
