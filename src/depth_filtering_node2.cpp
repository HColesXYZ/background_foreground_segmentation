#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <iostream>
#include <iterator>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
//#include <time.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sstream>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <pcl/filters/conditional_removal.h>

//std::string cloud_frame;
std::string output_topic;
int density_threshold;

uint num_cameras;
std::vector<sensor_msgs::CameraInfoConstPtr> cam_infos;
std::vector<std::string> image_frames;

sensor_msgs::ImageConstPtr image;

tf::TransformListener *tf_listener;
std::vector<ros::Publisher> publishers;

void printImageSizes(const sensor_msgs::ImageConstPtr& depth_image, const sensor_msgs::ImageConstPtr& mask_image) {
    // Get size of depth image
    int depth_width = depth_image->width;
    int depth_height = depth_image->height;

    // Get size of mask image
    int mask_width = mask_image->width;
    int mask_height = mask_image->height;

    ROS_INFO("Depth image size: %dx%d", depth_width, depth_height);
    ROS_INFO("Mask image size: %dx%d", mask_width, mask_height);
}

pcl::PointCloud<pcl::PointXYZI>
filter_pc2(const pcl::PointCloud<pcl::PointXYZI> &cloud,
          const sensor_msgs::ImageConstPtr &image,
          const sensor_msgs::CameraInfoConstPtr &c_info,
          const std::string &image_frame,
          const std::string &cloud_frame) {
  pcl::PointCloud<pcl::PointXYZI> out_cloud;

  std::shared_ptr<tf::StampedTransform> t(new tf::StampedTransform);
  try {
    tf_listener->lookupTransform(image_frame, cloud_frame, ros::Time(0), *t);
  } catch (const std::exception &e) {
    ROS_WARN("TF Failed %s", e.what());
    return out_cloud;
  }

  image_geometry::PinholeCameraModel model;
  model.fromCameraInfo(c_info);

  cv_bridge::CvImageConstPtr img;
  try {
    img = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    ROS_WARN("CVBridge Failed %s", e.what());
    return out_cloud;
  }
  const cv::Mat &camera_image = img->image;

  int num_points_in_image = 0;
  bool rectify_warning = false;
  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = cloud.begin();
       it != cloud.end(); ++it) {
    pcl::PointXYZI new_point;
    new_point.x = it->x;
    new_point.y = it->y;
    new_point.z = it->z;
    tf::Vector3 cloud_point(it->x, it->y, it->z);
    tf::Vector3 camera_local_tf = *t * cloud_point;
    if (camera_local_tf.z() <= 0)
      continue;
    cv::Point3d camera_local_cv(camera_local_tf.x(), camera_local_tf.y(),
                                camera_local_tf.z());
    cv::Point2d pixel = model.project3dToPixel(camera_local_cv);
    try {
      pixel = model.unrectifyPoint(pixel);
    } catch (const image_geometry::Exception &e) {
      if (!rectify_warning) {
        ROS_WARN("Could not unrectify image.");
        rectify_warning = true;
      }
    }
    if (pixel.x >= 0 && pixel.x < camera_image.cols && pixel.y >= 0 &&
        pixel.y < camera_image.rows) {
      uchar color = camera_image.at<uchar>(pixel.y, pixel.x);
      num_points_in_image += 1;
      if (color > density_threshold) {
        new_point.intensity = (float)std::max(color - density_threshold, 0);
        out_cloud.push_back(new_point);
      }
    }
  }
  ROS_INFO("Found %d points in image.", num_points_in_image);
  return out_cloud;
}

void _callback(const sensor_msgs::PointCloud2ConstPtr &cloud, const std::vector<sensor_msgs::ImageConstPtr> &images) {

    double startTime = ros::Time::now().toSec();

    for (uint i = 0; i < num_cameras; i++) {
        if (cam_infos[i] == nullptr) {
         ROS_INFO("Camera Calibration for Camera %d not ready yet", i);
         continue;
        }
        std::string iFrame = images[i]->header.frame_id;
        std::string cFrame = cloud->header.frame_id;

        ROS_INFO("Image Frame: %s, Cloud Frame: %s", iFrame.c_str(), cFrame.c_str());

        sensor_msgs::PointCloud2 out_cloud_msg;

        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(*cloud, pcl_cloud);
        pcl_cloud = filter_pc2(pcl_cloud, images[i], cam_infos[i], iFrame, cFrame);

        pcl::toROSMsg(pcl_cloud, out_cloud_msg);
        out_cloud_msg.header = cloud->header;
      
        publishers[i].publish(out_cloud_msg);

    }
    double timeDiff = ros::Time::now().toSec() - startTime;
    ROS_INFO("Published clouds in %.7fs, %.4f FPs for %d cams", timeDiff, 1 / timeDiff, num_cameras);
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud,
              const sensor_msgs::ImageConstPtr &image) {
    _callback(cloud, {image});
}

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud0,
              const sensor_msgs::PointCloud2ConstPtr &cloud1,
              const sensor_msgs::ImageConstPtr &image0,
              const sensor_msgs::ImageConstPtr &image1) {
  //_callback({cloud0 ,cloud1}, {image0, image1});
}

void setupPublishers(ros::NodeHandle& nh, const std::vector<std::string>& cloud_topics) {
    for (const auto& topic : cloud_topics) {
        std::string cloud_topic = topic + "/filtered"; // Appending "/filtered" to the image topic
        publishers.push_back(nh.advertise<sensor_msgs::PointCloud2>(cloud_topic, 1));
    }
}

void registerCamInfo(const sensor_msgs::CameraInfoConstPtr &c_info,
                     const uint index) {
  if (cam_infos.size() != num_cameras) {
    // initialize
    cam_infos.erase(cam_infos.begin(), cam_infos.end());
    for (uint i = 0; i < num_cameras; i++)
      cam_infos.push_back(nullptr);
  }
  cam_infos[index] = c_info;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "depth_filtering");
    ros::NodeHandle nh;
    std::vector<std::string> caminfo_topics;
    std::vector<std::string> cloud_topics;
    std::vector<std::string> mask_topics;

    ros::NodeHandle("~").getParam("cloud_topics", cloud_topics);
    ros::NodeHandle("~").getParam("mask_topics", mask_topics);
    ros::NodeHandle("~").getParam("caminfo_topics", caminfo_topics);
    ros::NodeHandle("~").getParam("density_threshold", density_threshold);

    std::ostringstream str;
    std::copy(cloud_topics.begin(), cloud_topics.end(), std::ostream_iterator<std::string>(str, ","));
    ROS_INFO("Depth Topics %s", str.str().c_str());
    num_cameras = cloud_topics.size();

    std::vector<ros::Subscriber> subs;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>>> cloud_subs;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> mask_subs;

    for (uint i = 0; i < num_cameras; i++) {
        cam_infos.push_back(nullptr);
        subs.push_back(nh.subscribe<sensor_msgs::CameraInfo>(caminfo_topics[i], 10, boost::bind(&registerCamInfo, _1, i)));
        cloud_subs.push_back( std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh, cloud_topics[i], 1));
        mask_subs.push_back( std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh, mask_topics[i], 1));
    }

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> ApproxSyncPolicy1;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy1>> sync1;

    sync1 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy1>>(ApproxSyncPolicy1(10), *cloud_subs[0], *mask_subs[0]);
    sync1->registerCallback(boost::bind(&callback, _1, _2));

    ROS_INFO("Registered callbacks for %d cameras.", num_cameras);

    setupPublishers(nh, cloud_topics);
    tf::TransformListener tl;
    tf_listener = &tl;
    ros::spin();
    return 0;
}
