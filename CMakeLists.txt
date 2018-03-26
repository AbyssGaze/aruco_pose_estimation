cmake_minimum_required(VERSION 2.8)
project(poseEstimation)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()



find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/3DTransformation.cpp
src/featureDetect.cpp
)
FILE(GLOB_RECURSE INC_ALL ${PROJECT_SOURCE_DIR}/include/*h)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)

# Build examples

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cpp ${INC_ALL})
target_link_libraries(rgbd_tum ${PROJECT_NAME} ${PCL_LIBRARIES})

add_executable(aruco_pose
Examples/RGB-D/aruco_pose.cpp ${INC_ALL})
target_link_libraries(aruco_pose ${PROJECT_NAME} ${PCL_LIBRARIES})

add_executable(seg_label
Examples/RGB-D/seg_label.cpp ${INC_ALL})
target_link_libraries(seg_label ${PROJECT_NAME} ${PCL_LIBRARIES})

add_executable(pcl_view
Examples/RGB-D/pcl_view.cpp ${INC_ALL})
target_link_libraries(pcl_view ${PROJECT_NAME} ${PCL_LIBRARIES})

add_executable(labeled_cloud_view
Examples/RGB-D/labeled_cloud_view.cpp ${INC_ALL})
target_link_libraries(labeled_cloud_view ${PROJECT_NAME} ${PCL_LIBRARIES})
