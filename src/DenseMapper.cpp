#include "DenseMapper.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace RigVis
{
	DenseCloud computeDenseCloud(const cv::Mat& leftImage, const cv::Mat& rightImage,
								 const StereoCalibration& calibration, const DenseCloudOptions& options,
								 const cv::Mat& colorImage)
	{
		DenseCloud cloud;
		if (leftImage.empty() || rightImage.empty()) return cloud;

		// Work on a scaled copy; intrinsics scale with the image, the baseline does not.
		cv::Mat left, right;
		if (options.imageScale != 1.0) {
			cv::resize(leftImage, left, cv::Size(), options.imageScale, options.imageScale, cv::INTER_AREA);
			cv::resize(rightImage, right, cv::Size(), options.imageScale, options.imageScale, cv::INTER_AREA);
		} else {
			left = leftImage;
			right = rightImage;
		}
		if (left.channels() == 3) cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
		if (right.channels() == 3) cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);

		const double fx = calibration.left.fx * options.imageScale;
		const double fy = calibration.left.fy * options.imageScale;
		const double cx = calibration.left.cx * options.imageScale;
		const double cy = calibration.left.cy * options.imageScale;
		const double focalTimesBaseline = fx * calibration.baseline_m;

		// Semi-global matching: P1/P2 penalise small/large disparity jumps between
		// neighbouring pixels (smoothness); uniqueness and speckle filters remove
		// ambiguous and isolated matches.
		const int block = options.blockSize;
		cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
			0, options.numDisparities, block, 8 * block * block, 32 * block * block,
			1, 63, 10, 100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
		cv::Mat disparity16;  // fixed point: disparity * 16
		sgbm->compute(left, right, disparity16);

		const double minDisparity = focalTimesBaseline / options.maxDepthMeters;
		const double maxDisparity = focalTimesBaseline / options.minDepthMeters;
		const int stride = std::max(1, options.pixelStride);

		cloud.pointsCamera.reserve(static_cast<std::size_t>(disparity16.total() / (stride * stride) / 2));
		cloud.intensity.reserve(cloud.pointsCamera.capacity());

		// Color camera: a different camera (~6 cm beside the left one on KITTI),
		// so each 3D point is projected into it rather than reusing (u, v).
		const bool useColor = calibration.has_color_camera && !colorImage.empty() && colorImage.type() == CV_8UC3;
		if (useColor) cloud.rgb.reserve(3 * cloud.pointsCamera.capacity());
		const Eigen::Matrix<double, 3, 4>& P = calibration.color_projection;

		for (int v = 0; v < disparity16.rows; v += stride) {
			const short* disparityRow = disparity16.ptr<short>(v);
			const std::uint8_t* intensityRow = left.ptr<std::uint8_t>(v);
			for (int u = 0; u < disparity16.cols; u += stride) {
				const double d = disparityRow[u] / 16.0;
				if (d < minDisparity || d > maxDisparity) continue;  // invalid, too far or too close

				const double z = focalTimesBaseline / d;
				const double x = (u - cx) * z / fx;
				const double y = (v - cy) * z / fy;
				cloud.pointsCamera.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
				cloud.intensity.push_back(intensityRow[u]);

				if (useColor) {
					// [u v 1] ~ P [X Y Z 1]: pixel in the full-resolution color image.
					const Eigen::Vector3d p = P * Eigen::Vector4d(x, y, z, 1.0);
					const int uc = static_cast<int>(std::lround(p.x() / p.z()));
					const int vc = static_cast<int>(std::lround(p.y() / p.z()));
					if (p.z() > 0.0 && uc >= 0 && uc < colorImage.cols && vc >= 0 && vc < colorImage.rows) {
						const cv::Vec3b bgr = colorImage.at<cv::Vec3b>(vc, uc);
						cloud.rgb.insert(cloud.rgb.end(), {bgr[2], bgr[1], bgr[0]});
					} else {
						// Outside the color image: fall back to gray, keeps the arrays aligned.
						const std::uint8_t g = intensityRow[u];
						cloud.rgb.insert(cloud.rgb.end(), {g, g, g});
					}
				}
			}
		}
		return cloud;
	}
}
