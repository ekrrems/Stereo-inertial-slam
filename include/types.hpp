#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace RigVis
{

	struct ImuSample
	{
		double timestamp_s = 0.0;

		// KITTI OXTS vehicle / IMU frame:
		// forward, left, up.
		Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
		Eigen::Vector3d angularVel = Eigen::Vector3d::Zero();
	};

	struct ImageFrame
	{
		double timestamp_s = 0.0;
		cv::Mat image;
	};

	struct StereoFrame
	{
		double timestamp_s = 0.0;

		ImageFrame left;
		ImageFrame right;

		// Optional color image (KITTI image_02, BGR). Empty if the dataset has none.
		// Only used to color the dense map; tracking uses the grayscale pair.
		cv::Mat color;
	};

	struct CameraCalibration
	{
		int width = 0;
		int height = 0;

		double fx = 0.0;
		double fy = 0.0;
		double cx = 0.0;
		double cy = 0.0;

		// KITTI sync images are already rectified.
		bool is_rectified = true;

		// Empty for rectified KITTI images.
		std::vector<double> distortion;
	};

	struct StereoCalibration
	{
		CameraCalibration left;
		CameraCalibration right;

		double baseline_m = 0.0;

		// Right camera coordinates → left camera coordinates.
		Eigen::Isometry3d left_from_right =
			Eigen::Isometry3d::Identity();

		// Color camera (KITTI P_rect_02): projects a point given in the rectified
		// LEFT camera frame into the color image: [u v 1]^T ~ P * [X Y Z 1]^T.
		bool has_color_camera = false;
		Eigen::Matrix<double, 3, 4> color_projection =
			Eigen::Matrix<double, 3, 4>::Zero();
	};

}  // namespace RigVis