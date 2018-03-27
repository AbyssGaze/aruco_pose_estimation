# aruco pose estimation
**Authors:** Y.Chen


# 1.Dependencies
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* C++11
* [catkin](http://wiki.ros.org/catkin) build system
* [opencv 3.1.0](https://github.com/opencv/opencv) compiled with stdC++11 enabled
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)


# 2. Building aruco_pose_estimation library and examples

```code
git clone http://chenying@git.zerozero.cn/chenying/aruco_pose_estimation.git
cd aruco_pose_estimation
mkdir build
cd build
cmake ..
make
```

# 3. Examples

## aruco_pose

Estimate current artag board pose and save it to a folder.

```
./aruco_pose ../param.yaml
```
* You can edit the camera intrinsic and artag board corners in **param.yaml**

## labeled_cloud_view

Visualization PointXYZL cloud result.
```
./labeled_cloud_view ../input.txt
```

## pcl_view

Convert depth and rgb file to point cloud.

```
./pcl_view ../one_object.txt
```
**one_object.txt** including depth image, rgb image and pose file name without postfix

## rgbd_tum

Convert depth and rgb file to a triangulated mesh cloud.

```
./rgbd_tum ../one_object.txt
```
**one_object.txt** including depth image, rgb image and pose file name without postfix

## seg_label

Building dataset of multiview segmentation using supervoxel segmentation and cloud render.

```
./seg_label ../input_label.txt
```
**input_label.txt** first row is the start image index

# Detail information you can find in html/index.html
