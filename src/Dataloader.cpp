#include "DataLoader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace
{

namespace fs = std::filesystem;

	std::vector<fs::path> sortedPngFiles(const fs::path& directory)
	{
		if (!fs::exists(directory)) {
			throw std::runtime_error(
				"Directory does not exist: " + directory.string());
		}

		std::vector<fs::path> files;

		for (const auto& entry : fs::directory_iterator(directory)) {
			if (entry.is_regular_file() &&
				entry.path().extension() == ".png")
			{
				files.push_back(entry.path());
			}
		}

		std::sort(files.begin(), files.end());

		return files;
	}

	double parseKittiTimestamp(const std::string& line)
	{
		int year = 0;
		int month = 0;
		int day = 0;
		int hour = 0;
		int minute = 0;
		double second = 0.0;

		const int parsed = std::sscanf(
			line.c_str(),
			"%d-%d-%d %d:%d:%lf",
			&year,
			&month,
			&day,
			&hour,
			&minute,
			&second);

		if (parsed != 6) {
			throw std::runtime_error(
				"Could not parse KITTI timestamp: " + line);
		}

		// One KITTI drive occurs on one day.
		// Relative seconds are enough for RigVis.
		return static_cast<double>(hour * 3600 + minute * 60) + second;
	}

	std::vector<double> readTimestamps(const fs::path& path)
	{
		std::ifstream file(path);

		if (!file.is_open()) {
			throw std::runtime_error(
				"Could not open timestamps file: " + path.string());
		}

		std::vector<double> timestamps;
		std::string line;

		while (std::getline(file, line)) {
			if (!line.empty()) {
				timestamps.push_back(parseKittiTimestamp(line));
			}
		}

		return timestamps;
	}

	std::unordered_map<std::string, std::vector<double>>
	readKittiCalibration(const fs::path& path)
	{
		std::ifstream file(path);

		if (!file.is_open()) {
			throw std::runtime_error(
				"Could not open calibration file: " + path.string());
		}

		std::unordered_map<std::string, std::vector<double>> values;

		std::string line;

		while (std::getline(file, line)) {
			std::istringstream stream(line);

			std::string key;
			stream >> key;

			if (key.empty()) {
				continue;
			}

			if (key.back() == ':') {
				key.pop_back();
			}

			std::vector<double> numbers;
			double value = 0.0;

			while (stream >> value) {
				numbers.push_back(value);
			}

			values[key] = std::move(numbers);
		}

		return values;
	}

	RigVis::ImuSample readOxtsSample(
		const fs::path& path,
		const double timestamp_s)
	{
		std::ifstream file(path);

		if (!file.is_open()) {
			throw std::runtime_error(
				"Could not open OXTS file: " + path.string());
		}

		std::vector<double> values;
		double value = 0.0;

		while (file >> value) {
			values.push_back(value);
		}

		// Need indices through wf, wl, wu.
		if (values.size() < 23) {
			throw std::runtime_error(
				"Invalid OXTS record: " + path.string());
		}

		RigVis::ImuSample sample;
		sample.timestamp_s = timestamp_s;

		// KITTI OXTS: ax, ay, az (11-13) and wx, wy, wz (17-19) are in the IMU
		// BODY frame, the frame calib_imu_to_velo.txt refers to (x forward,
		// y left, z up, gravity included: az ~ +9.8 at rest).
		// af/al/au and wf/wl/wu (14-16, 20-22) are in a different, vehicle-
		// aligned frame; with them the filter's tilt is pulled the wrong way
		// (tested on frames 0-160: r_rel 2.7 instead of 1.1 deg/100 m).
		sample.acceleration =
			Eigen::Vector3d(values[11], values[12], values[13]);

		sample.angularVel =
			Eigen::Vector3d(values[17], values[18], values[19]);

		return sample;
	}

}  // namespace

namespace RigVis
{

	DataLoader::DataLoader(const std::string& dataset_path, const std::string& imu_path, double imu_time_offset_s)
		: imu_time_offset_s_(imu_time_offset_s)
	{
		const fs::path drive_path(dataset_path);

		if (!fs::exists(drive_path)) {
			throw std::runtime_error(
				"KITTI drive path does not exist: " + drive_path.string());
		}

		// The parent day folder contains calib_cam_to_cam.txt.
		const fs::path day_path = drive_path.parent_path();

		loadCalibration(day_path);
		loadStereoRecords(drive_path);
		loadImuRecords(imu_path.empty() ? drive_path : fs::path(imu_path));

		if (stereo_frames_.empty()) {
			throw std::runtime_error("No KITTI stereo frames were found");
		}

		if (imu_samples_.empty()) {
			throw std::runtime_error("No KITTI OXTS/IMU samples were found");
		}

		// Read one image only to obtain width and height.
		const cv::Mat first_image = cv::imread(
			stereo_frames_.front().left_image_path.string(),
			cv::IMREAD_GRAYSCALE);

		if (first_image.empty()) {
			throw std::runtime_error(
				"Could not read first KITTI left image");
		}

		calibration_.left.width = first_image.cols;
		calibration_.left.height = first_image.rows;

		calibration_.right.width = first_image.cols;
		calibration_.right.height = first_image.rows;

		std::cout << "Loaded KITTI data:" << std::endl;
		std::cout << "  Stereo frames: " << stereo_frames_.size() << std::endl;
		std::cout << "  IMU samples: " << imu_samples_.size() << std::endl;
		std::cout << "  Baseline: " << calibration_.baseline_m << " m" << std::endl;
	}

	const StereoCalibration& DataLoader::calibration() const
	{
		return calibration_;
	}

	void DataLoader::loadCalibration(const fs::path& day_path)
	{
		const auto values = readKittiCalibration(
			day_path / "calib_cam_to_cam.txt");

		const auto left_iterator = values.find("P_rect_00");
		const auto right_iterator = values.find("P_rect_01");

		if (left_iterator == values.end() ||
			right_iterator == values.end())
		{
			throw std::runtime_error(
				"Could not find P_rect_00 / P_rect_01 in calib_cam_to_cam.txt");
		}

		const auto& left_projection = left_iterator->second;
		const auto& right_projection = right_iterator->second;

		if (left_projection.size() != 12 ||
			right_projection.size() != 12)
		{
			throw std::runtime_error(
				"KITTI projection matrices must have 12 values");
		}

		calibration_.left.fx = left_projection[0];
		calibration_.left.fy = left_projection[5];
		calibration_.left.cx = left_projection[2];
		calibration_.left.cy = left_projection[6];

		calibration_.right.fx = right_projection[0];
		calibration_.right.fy = right_projection[5];
		calibration_.right.cx = right_projection[2];
		calibration_.right.cy = right_projection[6];

		// For a rectified pair:
		// P = [fx 0 cx tx]
		// camera center x = tx / fx
		const double left_center_x =
			left_projection[3] / left_projection[0];

		const double right_center_x =
			right_projection[3] / right_projection[0];

		calibration_.baseline_m =
			std::abs(right_center_x - left_center_x);

		if (calibration_.baseline_m <= 0.0) {
			throw std::runtime_error(
				"Invalid KITTI stereo baseline");
		}

		// Right-camera coordinates → left-camera coordinates.
		calibration_.left_from_right =
			Eigen::Isometry3d::Identity();

		calibration_.left_from_right.translation().x() =
			calibration_.baseline_m;

		// Color camera projection (used only if image_02 exists, see loadStereoRecords).
		const auto color_iterator = values.find("P_rect_02");
		if (color_iterator != values.end() && color_iterator->second.size() == 12) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 4; ++column) {
					calibration_.color_projection(row, column) =
						color_iterator->second[static_cast<std::size_t>(row * 4 + column)];
				}
			}
		}

		// IMU -> rectified left camera, chained from the KITTI calibration files:
		//   x_velo = T_velo_imu x_imu,  x_cam0 = T_cam0_velo x_velo,  x_rect = R_rect_00 x_cam0
		const auto rect_iterator = values.find("R_rect_00");
		const fs::path imu_to_velo_path = day_path / "calib_imu_to_velo.txt";
		const fs::path velo_to_cam_path = day_path / "calib_velo_to_cam.txt";
		if (rect_iterator != values.end() && rect_iterator->second.size() == 9 && fs::exists(imu_to_velo_path) &&
			fs::exists(velo_to_cam_path)) {
			auto rigid = [](const std::unordered_map<std::string, std::vector<double>>& file) {
				Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
				const auto& R = file.at("R");
				const auto& T = file.at("T");
				for (int row = 0; row < 3; ++row) {
					for (int column = 0; column < 3; ++column) {
						transform.linear()(row, column) = R[static_cast<std::size_t>(row * 3 + column)];
					}
					transform.translation()(row) = T[static_cast<std::size_t>(row)];
				}
				return transform;
			};
			Eigen::Isometry3d rectFromCam0 = Eigen::Isometry3d::Identity();
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					rectFromCam0.linear()(row, column) = rect_iterator->second[static_cast<std::size_t>(row * 3 + column)];
				}
			}
			calibration_.camera_from_imu = rectFromCam0 * rigid(readKittiCalibration(velo_to_cam_path)) *
										   rigid(readKittiCalibration(imu_to_velo_path));
			calibration_.has_imu_extrinsics = true;
		}
	}

	void DataLoader::loadStereoRecords(const fs::path& drive_path)
	{
		const fs::path left_data_path =
			drive_path / "image_00" / "data";

		const fs::path right_data_path =
			drive_path / "image_01" / "data";

		const std::vector<fs::path> left_images =
			sortedPngFiles(left_data_path);

		const std::vector<fs::path> right_images =
			sortedPngFiles(right_data_path);

		const std::vector<double> left_timestamps =
			readTimestamps(drive_path / "image_00" / "timestamps.txt");

		const std::vector<double> right_timestamps =
			readTimestamps(drive_path / "image_01" / "timestamps.txt");

		if (left_images.size() != right_images.size() ||
			left_images.size() != left_timestamps.size() ||
			left_images.size() != right_timestamps.size())
		{
			throw std::runtime_error(
				"KITTI left/right image counts or timestamp counts differ");
		}

		stereo_frames_.reserve(left_images.size());

		for (std::size_t index = 0; index < left_images.size(); ++index) {
			// const double timestamp_difference =
			// 	std::abs(left_timestamps[index] - right_timestamps[index]);

			// if (timestamp_difference > 0.001) {
			// 	throw std::runtime_error(
			// 		"KITTI left/right timestamps are not synchronized");
			// }

			const double time_difference_s =
				std::abs(left_timestamps[index] - right_timestamps[index]);

			if (time_difference_s > 0.01) {
				throw std::runtime_error(
					"KITTI stereo timestamp mismatch at frame " +
					std::to_string(index) +
					", dt=" + std::to_string(time_difference_s) + " s");
			}

			StereoRecord record;
			record.timestamp_s =
				0.5 * (left_timestamps[index] + right_timestamps[index]);
			// record.timestamp_s = left_timestamps[index];
			record.left_image_path = left_images[index];
			record.right_image_path = right_images[index];

			stereo_frames_.push_back(std::move(record));
		}

		// Optional color camera: image_02 with one image per stereo frame.
		const fs::path color_data_path = drive_path / "image_02" / "data";
		if (fs::exists(color_data_path) && !calibration_.color_projection.isZero()) {
			const std::vector<fs::path> color_images = sortedPngFiles(color_data_path);
			if (color_images.size() == stereo_frames_.size()) {
				for (std::size_t index = 0; index < color_images.size(); ++index) {
					stereo_frames_[index].color_image_path = color_images[index];
				}
				calibration_.has_color_camera = true;
				std::cout << "Color camera found (image_02): dense map will be colored\n";
			} else {
				std::cout << "image_02 has " << color_images.size() << " images, expected "
						  << stereo_frames_.size() << ": ignoring color\n";
			}
		}
	}

	void DataLoader::loadImuRecords(const fs::path& drive_path)
	{
		const fs::path oxts_data_path =
			drive_path / "oxts" / "data";

		// const std::vector<fs::path> oxts_files =
		// 	sortedPngFiles(oxts_data_path);

		// OXTS data uses .txt, not .png.
		std::vector<fs::path> imu_files;

		for (const auto& entry : fs::directory_iterator(oxts_data_path)) {
			if (entry.is_regular_file() &&
				entry.path().extension() == ".txt")
			{
				imu_files.push_back(entry.path());
			}
		}

		std::sort(imu_files.begin(), imu_files.end());

		const std::vector<double> timestamps =
			readTimestamps(drive_path / "oxts" / "timestamps.txt");

		if (imu_files.size() != timestamps.size()) {
			throw std::runtime_error(
				"KITTI OXTS file count and timestamp count differ");
		}

		imu_samples_.reserve(imu_files.size());

		for (std::size_t index = 0; index < imu_files.size(); ++index) {
			imu_samples_.push_back(
				readOxtsSample(imu_files[index], timestamps[index] + imu_time_offset_s_));
		}
	}

	StereoFrame DataLoader::loadStereoFrame(
		const StereoRecord& record) const
	{
		StereoFrame frame;
		frame.timestamp_s = record.timestamp_s;

		frame.left.timestamp_s = record.timestamp_s;
		frame.left.image = cv::imread(
			record.left_image_path.string(),
			cv::IMREAD_GRAYSCALE);

		frame.right.timestamp_s = record.timestamp_s;
		frame.right.image = cv::imread(
			record.right_image_path.string(),
			cv::IMREAD_GRAYSCALE);

		if (frame.left.image.empty() || frame.right.image.empty()) {
			throw std::runtime_error(
				"Could not load KITTI stereo image pair");
		}

		if (!record.color_image_path.empty()) {
			frame.color = cv::imread(record.color_image_path.string(), cv::IMREAD_COLOR);
		}

		return frame;
	}

	bool DataLoader::next(SensorEvent& event)
	{
		const bool imu_finished =
			next_imu_index_ >= imu_samples_.size();

		const bool stereo_finished =
			next_stereo_index_ >= stereo_frames_.size();

		if (imu_finished && stereo_finished) {
			return false;
		}

		// At equal timestamps, give IMU to RigVis first.
		if (!imu_finished &&
			(stereo_finished ||
			imu_samples_[next_imu_index_].timestamp_s <=
			stereo_frames_[next_stereo_index_].timestamp_s))
		{
			event = imu_samples_[next_imu_index_];
			++next_imu_index_;

			return true;
		}

		event = loadStereoFrame(
			stereo_frames_[next_stereo_index_]);

		++next_stereo_index_;

		return true;
	}

}  // namespace RigVis