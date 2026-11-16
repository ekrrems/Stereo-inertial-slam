/* RigVis map export: binary PLY point clouds + keyframe CSV.

   PLY (x y z float, red green blue uchar) opens in Metashape (File > Import >
   Import Points), CloudCompare, MeshLab and Blender.

   World frame: the first camera. Camera convention is x right, y down,
   z forward, so viewers show the map upside down / lying on its side. With
   zUp = true every point is converted to (x, z, -y): x right, y forward, z up.
*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>  // applyColorMap (Turbo colormap)

#include "StereoFrontend.hpp"

namespace RigVis
{
	namespace
	{
		struct PlyVertex
		{
			float x, y, z;
			std::uint8_t r, g, b;
		};

		Eigen::Matrix3d exportRotation(bool zUp)
		{
			Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
			if (zUp) rotation << 1, 0, 0,  //  x stays right
								 0, 0, 1,  //  y_out = z (forward)
								 0, -1, 0; //  z_out = -y (up)
			return rotation;
		}

		bool writePly(const std::string& path, const std::vector<PlyVertex>& vertices)
		{
			std::ofstream file(path, std::ios::binary);
			if (!file) {
				std::cout << "Could not write " << path << '\n';
				return false;
			}
			file << "ply\nformat binary_little_endian 1.0\n"
				 << "comment RigVis stereo SLAM map\n"
				 << "element vertex " << vertices.size() << '\n'
				 << "property float x\nproperty float y\nproperty float z\n"
				 << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
				 << "end_header\n";
			for (const PlyVertex& v : vertices) {  // write field by field: no struct padding in the file
				file.write(reinterpret_cast<const char*>(&v.x), sizeof(float));
				file.write(reinterpret_cast<const char*>(&v.y), sizeof(float));
				file.write(reinterpret_cast<const char*>(&v.z), sizeof(float));
				file.write(reinterpret_cast<const char*>(&v.r), 3);
			}
			std::cout << "Wrote " << vertices.size() << " points to " << path << '\n';
			return true;
		}

		PlyVertex makeVertex(const Eigen::Vector3d& p, std::uint8_t r, std::uint8_t g, std::uint8_t b)
		{
			return {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()), r, g, b};
		}
	}

	bool StereoFrontend::exportSparseMapPly(const std::string& path, bool zUp) const
	{
		const Eigen::Matrix3d rotation = exportRotation(zUp);
		std::vector<PlyVertex> vertices;
		vertices.reserve(mapPoints_.size());
		for (const SparseMapPoint& point : mapPoints_) {
			if (!point.valid) continue;
			vertices.push_back(makeVertex(rotation * point.positionWorld, point.intensity, point.intensity,
										  point.intensity));
		}
		return writePly(path, vertices);
	}

	// All dense keyframe clouds, transformed with the CURRENT keyframe poses and
	// merged on a voxel grid (one averaged point per voxel) to keep the file small.
	bool StereoFrontend::exportDenseMapPly(const std::string& path, double voxelSizeMeters, bool zUp,
										   DenseColorMode colorMode) const
	{
		struct Voxel
		{
			Eigen::Vector3d sum = Eigen::Vector3d::Zero();
			Eigen::Vector3d rgbSum = Eigen::Vector3d::Zero();
			double intensitySum = 0.0;
			int count = 0;
		};

		bool anyRgb = false;
		for (const auto& entry : denseClouds_) anyRgb = anyRgb || !entry.second.rgb.empty();
		if (colorMode == DenseColorMode::Auto) colorMode = anyRgb ? DenseColorMode::Rgb : DenseColorMode::Height;
		if (colorMode == DenseColorMode::Rgb && !anyRgb) {
			std::cout << "No color images were available: using height coloring instead of RGB\n";
			colorMode = DenseColorMode::Height;
		}
		const Eigen::Matrix3d rotation = exportRotation(zUp);
		const double inverseVoxel = 1.0 / voxelSizeMeters;
		std::unordered_map<std::uint64_t, Voxel> voxels;
		std::size_t rawPoints = 0;

		auto key = [&](const Eigen::Vector3d& p) {
			constexpr std::int64_t offset = 1 << 20;  // 21 bits per axis: +-1M voxels
			const auto cell = [&](double v) {
				return static_cast<std::uint64_t>((static_cast<std::int64_t>(std::floor(v * inverseVoxel)) + offset) &
												  0x1FFFFF);
			};
			return (cell(p.x()) << 42) | (cell(p.y()) << 21) | cell(p.z());
		};

		for (const auto& [keyframeId, cloud] : denseClouds_) {
			if (keyframeId >= keyframes_.size()) continue;
			const Eigen::Isometry3d& worldFromCamera = keyframes_[keyframeId].worldFromCamera;
			for (std::size_t i = 0; i < cloud.pointsCamera.size(); ++i) {
				const Eigen::Vector3d world = rotation * (worldFromCamera * cloud.pointsCamera[i].cast<double>());
				Voxel& voxel = voxels[key(world)];
				voxel.sum += world;
				voxel.intensitySum += cloud.intensity[i];
				if (!cloud.rgb.empty()) {
					voxel.rgbSum += Eigen::Vector3d(cloud.rgb[3 * i], cloud.rgb[3 * i + 1], cloud.rgb[3 * i + 2]);
				} else {
					voxel.rgbSum += Eigen::Vector3d::Constant(cloud.intensity[i]);  // gray for clouds without color
				}
				++voxel.count;
				++rawPoints;
			}
		}

		std::vector<PlyVertex> vertices;
		vertices.reserve(voxels.size());
		for (const auto& [k, voxel] : voxels) {
			const Eigen::Vector3d position = voxel.sum / voxel.count;
			if (colorMode == DenseColorMode::Rgb) {
				const Eigen::Vector3d rgb = voxel.rgbSum / voxel.count;
				vertices.push_back(makeVertex(position, static_cast<std::uint8_t>(rgb.x()),
											  static_cast<std::uint8_t>(rgb.y()), static_cast<std::uint8_t>(rgb.z())));
			} else {
				const auto gray = static_cast<std::uint8_t>(voxel.intensitySum / voxel.count);
				vertices.push_back(makeVertex(position, gray, gray, gray));
			}
		}

		// Height coloring: Turbo colormap (blue = low ... red = high) over the
		// 2nd..98th height percentile, so a few outliers do not squash the scale.
		if (colorMode == DenseColorMode::Height && !vertices.empty()) {
			auto height = [&](const PlyVertex& v) { return zUp ? v.z : -v.y; };  // camera frame: up = -y
			std::vector<float> heights;
			heights.reserve(vertices.size());
			for (const PlyVertex& v : vertices) heights.push_back(height(v));
			std::sort(heights.begin(), heights.end());
			const float low = heights[heights.size() * 2 / 100];
			const float high = heights[std::min(heights.size() - 1, heights.size() * 98 / 100)];
			const float range = std::max(high - low, 1e-3F);

			cv::Mat ramp(1, 256, CV_8UC1), lut;
			for (int i = 0; i < 256; ++i) ramp.at<std::uint8_t>(0, i) = static_cast<std::uint8_t>(i);
			cv::applyColorMap(ramp, lut, cv::COLORMAP_TURBO);  // 256 BGR colors

			for (PlyVertex& v : vertices) {
				const float t = std::clamp((height(v) - low) / range, 0.0F, 1.0F);
				const cv::Vec3b bgr = lut.at<cv::Vec3b>(0, static_cast<int>(t * 255.0F));
				v.r = bgr[2];
				v.g = bgr[1];
				v.b = bgr[0];
			}
			std::cout << "Height colors: " << low << " m (blue) .. " << high << " m (red)\n";
		}

		const char* modeName = colorMode == DenseColorMode::Rgb ? "rgb" : colorMode == DenseColorMode::Height ? "height" : "gray";
		std::cout << "Dense map: " << denseClouds_.size() << " keyframe clouds, " << rawPoints << " points -> "
				  << vertices.size() << " voxels of " << voxelSizeMeters << " m, colors: " << modeName << '\n';
		return writePly(path, vertices);
	}

	// Keyframe camera centres (red) plus a point 1 m along each viewing direction (yellow).
	bool StereoFrontend::exportCamerasPly(const std::string& path, bool zUp) const
	{
		const Eigen::Matrix3d rotation = exportRotation(zUp);
		std::vector<PlyVertex> vertices;
		for (const Keyframe& keyframe : keyframes_) {
			const Eigen::Vector3d centre = keyframe.worldFromCamera.translation();
			const Eigen::Vector3d ahead = keyframe.worldFromCamera * Eigen::Vector3d(0, 0, 1.0);
			vertices.push_back(makeVertex(rotation * centre, 255, 0, 0));
			vertices.push_back(makeVertex(rotation * ahead, 255, 220, 0));
		}
		return writePly(path, vertices);
	}

	// One row per keyframe. Position is in the (optionally z-up) map frame.
	// Orientation as omega/phi/kappa [deg] with the classic photogrammetric
	// convention (R_world_to_camera = Rx(omega) * Ry(phi) * Rz(kappa), camera
	// x right, y up, z backwards). Positions import reliably; check the angle
	// convention in Metashape's reference import dialog before trusting them.
	bool StereoFrontend::exportKeyframesCsv(const std::string& path, bool zUp) const
	{
		std::ofstream csv(path);
		if (!csv) {
			std::cout << "Could not write " << path << '\n';
			return false;
		}
		const Eigen::Matrix3d mapRotation = exportRotation(zUp);
		// OpenCV camera (x right, y down, z forward) -> photogrammetric camera (x right, y up, z back)
		const Eigen::Matrix3d toPhotogrammetric = Eigen::Vector3d(1, -1, -1).asDiagonal();
		constexpr double radToDeg = 57.29577951308232;

		csv << "label,frame,x,y,z,omega,phi,kappa,qx,qy,qz,qw\n";
		csv << std::fixed << std::setprecision(6);
		for (const Keyframe& keyframe : keyframes_) {
			const Eigen::Vector3d position = mapRotation * keyframe.worldFromCamera.translation();
			const Eigen::Matrix3d worldFromCamera = mapRotation * keyframe.worldFromCamera.linear();
			const Eigen::Matrix3d worldToCamera = toPhotogrammetric * worldFromCamera.transpose();

			// R = Rx(omega) Ry(phi) Rz(kappa): phi = asin(R(0,2)), omega and kappa from the rest.
			const double phi = std::asin(std::clamp(worldToCamera(0, 2), -1.0, 1.0));
			const double omega = std::atan2(-worldToCamera(1, 2), worldToCamera(2, 2));
			const double kappa = std::atan2(-worldToCamera(0, 1), worldToCamera(0, 0));
			const Eigen::Quaterniond q(worldFromCamera);

			std::ostringstream label;
			label << std::setw(10) << std::setfill('0') << keyframe.frameIndex << ".png";
			csv << label.str() << ',' << keyframe.frameIndex << ',' << position.x() << ',' << position.y() << ','
				<< position.z() << ',' << omega * radToDeg << ',' << phi * radToDeg << ',' << kappa * radToDeg << ','
				<< q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
		}
		std::cout << "Wrote " << keyframes_.size() << " keyframes to " << path << '\n';
		return true;
	}
}
