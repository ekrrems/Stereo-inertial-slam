/* RigVis dense stereo point clouds per keyframe.

   For a keyframe's rectified stereo pair, cv::StereoSGBM computes a disparity
   for (almost) every pixel; each valid pixel becomes a 3D point:
       Z = fx * b / d,   X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy
   Points are stored in the KEYFRAME'S CAMERA FRAME, so bundle adjustment and
   loop closure move them automatically with their keyframe (world point =
   keyframe pose * camera point at export time). The same clouds are the input
   for ICP later.

   Depth noise grows with Z^2 (sigma_Z = Z^2 / (fx b) * sigma_d), so only the
   near range is kept (maxDepthMeters).
*/
#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "types.hpp"

namespace RigVis
{
	struct DenseCloudOptions
	{
		bool enabled = true;
		std::size_t everyNthKeyframe = 2;  // dense cloud for every Nth keyframe (memory/time)
		double imageScale = 0.5;           // SGBM on half resolution: ~4x faster
		int numDisparities = 96;           // at the scaled resolution, multiple of 16 (-> min depth ~2 m)
		int blockSize = 5;
		int pixelStride = 2;               // keep every Nth pixel of the scaled disparity image
		double maxDepthMeters = 20.0;      // beyond this stereo depth is too noisy
		double minDepthMeters = 1.0;
	};

	struct DenseCloud
	{
		std::vector<Eigen::Vector3f> pointsCamera;  // in the keyframe's left camera frame
		std::vector<std::uint8_t> intensity;        // gray value from the left image
		std::vector<std::uint8_t> rgb;              // R,G,B per point from the color camera; empty if none
	};

	// colorImage is optional (empty = no color). Each point is projected into it
	// with calibration.color_projection (KITTI P_rect_02) to pick its color.
	DenseCloud computeDenseCloud(const cv::Mat& leftImage, const cv::Mat& rightImage,
								 const StereoCalibration& calibration, const DenseCloudOptions& options,
								 const cv::Mat& colorImage = cv::Mat());
}
