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

std::string cloud_frame;
std::string output_topic;
int density_threshold;

uint num_cameras;
std::vector<sensor_msgs::CameraInfoConstPtr> cam_infos;
std::vector<std::string> image_frames;

sensor_msgs::ImageConstPtr image;

tf::TransformListener *tf_listener;
//ros::Publisher *publisher;
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

sensor_msgs::ImageConstPtr filter_depth(const sensor_msgs::ImageConstPtr& depth_image,
                                        const sensor_msgs::ImageConstPtr& mask_image) {
    // Check if input images are valid
    if (depth_image == nullptr || mask_image == nullptr) {
        ROS_WARN("Received null depth or mask image!");
        return nullptr;
    }

    // Convert depth image to OpenCV format for processing
    cv_bridge::CvImageConstPtr cv_depth_ptr;
    try {
        cv_depth_ptr = cv_bridge::toCvShare(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return nullptr;
    }

    // Convert mask image to OpenCV format for processing
    cv_bridge::CvImageConstPtr cv_mask_ptr;
    try {
        cv_mask_ptr = cv_bridge::toCvShare(mask_image, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return nullptr;
    }

    // Create output image
    cv_bridge::CvImage cv_filtered_image;
    cv_filtered_image.header = depth_image->header;
    cv_filtered_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_filtered_image.image = cv::Mat::zeros(cv_depth_ptr->image.size(), CV_32FC1);

    // Loop through each pixel of the depth image
    for (int y = 0; y < cv_depth_ptr->image.rows; ++y) {
        for (int x = 0; x < cv_depth_ptr->image.cols; ++x) {
            // Check if the corresponding pixel in the mask is above the density threshold
            if (cv_mask_ptr->image.at<uchar>(y, x) > density_threshold) {
                // Copy the depth value to the filtered image
                cv_filtered_image.image.at<float>(y, x) = cv_depth_ptr->image.at<float>(y, x);
                //ROS_INFO("Camera Calibration for Camera %d not ready yet", cv_mask_ptr->image.at<uchar>(y, x));
                //ROS_INFO("Threshold: %d", density_threshold);
            }
            //else{
                //ROS_INFO("Camera Calibration for Camera %d not ready yet", cv_mask_ptr->image.at<uchar>(y, x));
                //ROS_INFO("Threshold: %d", density_threshold);
            //}
        }
    }
    // Convert filtered image back to ROS format
    sensor_msgs::ImagePtr filtered_image_msg = cv_filtered_image.toImageMsg();

    return filtered_image_msg;
}

void setupPublishers(ros::NodeHandle& nh, const std::vector<std::string>& image_topics) {
    for (const auto& topic : image_topics) {
        std::string cloud_topic = topic + "/cloud"; // Appending "/cloud" to the image topic
        publishers.push_back(nh.advertise<sensor_msgs::PointCloud2>(cloud_topic, 1));
    }
}

pcl::PointCloud<pcl::PointXYZ> depth_2_pcl(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &c_info) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    // Check if the image is valid
    if (image->data.empty() || c_info->K.empty()) {
        ROS_WARN("Empty image or camera info.");
        return cloud;
    }

    // Assuming the depth image is of type 32FC1
    if (image->encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
        ROS_WARN("Depth image encoding is not 32FC1.");
        return cloud;
    }

    // Retrieve camera parameters
    double fx = c_info->K[0];
    double fy = c_info->K[4];
    double cx = c_info->K[2];
    double cy = c_info->K[5];

    // Convert depth image to point cloud
    for (int v = 0; v < image->height; ++v) {
        for (int u = 0; u < image->width; ++u) {
            float depth = *reinterpret_cast<const float*>(&image->data[(v * image->width + u) * 4]); // Assuming little endian

            if (std::isnan(depth) || depth <= 0.0) {
                continue;
            }

            pcl::PointXYZ point;
            point.x = (u - cx) * depth / fx;
            point.y = (v - cy) * depth / fy;
            point.z = depth;
            //point.intensity = 0.0; // You may set this according to your requirement

            cloud.push_back(point);
        }
    }

    // Set the frame ID of the point cloud
    //cloud.header = image->header;
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = false; // Some points might be NaN, hence not dense
    return cloud;
}

void _callback(const std::vector<sensor_msgs::ImageConstPtr> &images,
               const std::vector<sensor_msgs::ImageConstPtr> &masks) {
  
  double startTime = ros::Time::now().toSec();

  for (uint i = 0; i < num_cameras; i++) {
    // check if caminfo ready
    if (cam_infos[i] == nullptr) {
      ROS_INFO("Camera Calibration for Camera %d not ready yet", i);
    } 
    else {
      sensor_msgs::PointCloud2Ptr out_cloud_msg = sensor_msgs::PointCloud2Ptr(new sensor_msgs::PointCloud2());

      pcl::PointCloud<pcl::PointXYZ> out_cloud = depth_2_pcl(filter_depth(images[i], masks[i]), cam_infos[i]);

      pcl::toROSMsg(out_cloud, *out_cloud_msg);
      out_cloud_msg->header = images[i]->header;
      publishers[i].publish(out_cloud_msg);
    }
  }

  double timeDiff = ros::Time::now().toSec() - startTime;
  ROS_INFO("Published clouds in %.7fs, %.4f FPs for %d cams", timeDiff, 1 / timeDiff, num_cameras);
}

void callback(const sensor_msgs::ImageConstPtr& image0,
              const sensor_msgs::ImageConstPtr& image1,
              const sensor_msgs::ImageConstPtr& image2,
              const sensor_msgs::ImageConstPtr& image3,
              const sensor_msgs::ImageConstPtr& mask0,
              const sensor_msgs::ImageConstPtr& mask1,
              const sensor_msgs::ImageConstPtr& mask2,
              const sensor_msgs::ImageConstPtr& mask3) {
  _callback({image0, image1, image2, image3}, {mask0, mask1, mask2, mask3});              
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
    std::vector<std::string> image_topics;
    std::vector<std::string> mask_topics;

    ros::NodeHandle("~").getParam("image_topics", image_topics);
    ros::NodeHandle("~").getParam("mask_topics", mask_topics);
    ros::NodeHandle("~").getParam("caminfo_topics", caminfo_topics);
    ros::NodeHandle("~").getParam("density_threshold", density_threshold);

    std::ostringstream str;
    std::copy(image_topics.begin(), image_topics.end(), std::ostream_iterator<std::string>(str, ","));
    ROS_INFO("Depth Topics %s", str.str().c_str());
    num_cameras = image_topics.size();

    std::vector<ros::Subscriber> subs;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> image_subs;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> mask_subs;

    for (uint i = 0; i < num_cameras; i++) {
        subs.push_back(nh.subscribe<sensor_msgs::CameraInfo>(
            caminfo_topics[i], 10, boost::bind(&registerCamInfo, _1, i)));
        image_subs.push_back(
            std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
                nh, image_topics[i], 1));
        mask_subs.push_back(
            std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
                nh, mask_topics[i], 1));
    }

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                            sensor_msgs::Image, sensor_msgs::Image,
                                                            sensor_msgs::Image, sensor_msgs::Image,
                                                            sensor_msgs::Image, sensor_msgs::Image> ApproxSyncPolicy8;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy8>> sync8;

    sync8 = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy8>>(
        ApproxSyncPolicy8(10), *image_subs[0], *image_subs[1], *image_subs[2], *image_subs[3],
        *mask_subs[0], *mask_subs[1], *mask_subs[2], *mask_subs[3]);
    sync8->registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6, _7, _8));

    ROS_INFO("Registered callbacks for %d cameras.", num_cameras);

    setupPublishers(nh, image_topics);
    tf::TransformListener tl;
    tf_listener = &tl;
    ros::spin();
    return 0;
}
