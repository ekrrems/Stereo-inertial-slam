#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "types.hpp"

namespace RigVis
{

	using SensorEvent = std::variant<ImuSample, StereoFrame>;

	class DataLoader
	{
		public:
			// dataset_path: the exact *_sync drive directory.
			// imu_path: optional *_extract drive directory with 100 Hz OXTS/IMU data.
			//           Empty = use the 10 Hz OXTS data inside the *_sync directory.
			// imu_time_offset_s is added to every IMU timestamp (the OXTS clock lags
			// the cameras; about -0.05 s on 2011_10_03_drive_0027, see config).
			explicit DataLoader(const std::string& dataset_path, const std::string& imu_path = "",
								double imu_time_offset_s = 0.0);

			// Returns false after every IMU sample and stereo pair is consumed.
			bool next(SensorEvent& event);

			const StereoCalibration& calibration() const;

		private:
			struct StereoRecord
			{
				double timestamp_s = 0.0;

				std::filesystem::path left_image_path;
				std::filesystem::path right_image_path;
				std::filesystem::path color_image_path;  // empty if image_02 is not available
			};

			void loadCalibration(const std::filesystem::path& day_path);

			void loadStereoRecords(
				const std::filesystem::path& drive_path);

			void loadImuRecords(
				const std::filesystem::path& drive_path);

			StereoFrame loadStereoFrame(
				const StereoRecord& record) const;

			StereoCalibration calibration_;

			std::vector<StereoRecord> stereo_frames_;
			std::vector<ImuSample> imu_samples_;

			std::size_t next_stereo_index_ = 0;
			std::size_t next_imu_index_ = 0;
			double imu_time_offset_s_ = 0.0;
	};

}  // namespace RigVis