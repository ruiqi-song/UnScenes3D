cmake_minimum_required(VERSION 3.5)
project(uns_label4d)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MAJOR_VERSION "1")
set(MINOR_VERSION "0")
set(PATCH_VERSION "0")

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  rospy
  roscpp
  tf
  tf2
  tf2_eigen
  tf2_ros
  std_msgs
  sensor_msgs
  autoware_msgs
  autoware_config_msgs
  jsk_recognition_msgs
)
catkin_package()

include_directories(
  common
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(obs_4d_builder
    common/cluster_polygon.cpp
    obs_4d_builder.cpp)
target_link_libraries(obs_4d_builder ${catkin_LIBRARIES} ${PCL_LIBRARIES})
