#include "Evaluation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <Eigen/Geometry>  // Eigen::umeyama
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace RigVis
{
	namespace
	{
		constexpr double kPi = 3.14159265358979323846;
		constexpr double kRadToDeg = 180.0 / kPi;

		// KITTI devkit (evaluate_odometry.cpp) values.
		const std::vector<double> kSegmentLengths = {100, 200, 300, 400, 500, 600, 700, 800};
		constexpr std::size_t kStartFrameStep = 10;

		// "KEY: v1 v2 ..." lines -> map. Non-numeric lines (calib_time) are skipped.
		std::map<std::string, std::vector<double>> readCalibrationFile(const fs::path& path)
		{
			std::ifstream file(path);
			if (!file) throw std::runtime_error("Could not open calibration file: " + path.string());

			std::map<std::string, std::vector<double>> values;
			std::string line;
			while (std::getline(file, line)) {
				const auto colon = line.find(':');
				if (colon == std::string::npos) continue;

				std::istringstream numbers(line.substr(colon + 1));
				std::vector<double> parsed;
				double value = 0.0;
				while (numbers >> value) parsed.push_back(value);
				if (!parsed.empty()) values[line.substr(0, colon)] = parsed;
			}
			return values;
		}

		const std::vector<double>& requireKey(const std::map<std::string, std::vector<double>>& values,
											  const std::string& key, std::size_t count,
											  const fs::path& path)
		{
			const auto it = values.find(key);
			if (it == values.end() || it->second.size() != count) {
				throw std::runtime_error("Missing or malformed '" + key + "' in " + path.string());
			}
			return it->second;
		}

		// Nearest true rotation (SVD: R = U V^T). Calibration and pose files store
		// rotations with 7-9 digits, so they are slightly non-orthonormal. Eigen's
		// Isometry3d::inverse() assumes R^-1 = R^T, and the leftover ~1e-6 error
		// inflates near-zero rotation angles (acos near 1) by ~0.1 deg.
		Eigen::Matrix3d orthonormalize(const Eigen::Matrix3d& matrix)
		{
			const Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
			if (rotation.determinant() < 0.0) {
				Eigen::Matrix3d u = svd.matrixU();
				u.col(2) *= -1.0;
				rotation = u * svd.matrixV().transpose();
			}
			return rotation;
		}

		Eigen::Isometry3d makeRigid(const std::vector<double>& rotationRowMajor,
									const std::vector<double>& translation)
		{
			Eigen::Matrix3d rotation;
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					rotation(row, column) = rotationRowMajor[static_cast<std::size_t>(row * 3 + column)];
				}
			}
			Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
			transform.linear() = orthonormalize(rotation);
			if (!translation.empty()) {
				transform.translation() = Eigen::Vector3d(translation[0], translation[1], translation[2]);
			}
			return transform;
		}

		double rotationAngle(const Eigen::Matrix3d& rotation)
		{
			const double cosine = std::clamp((rotation.trace() - 1.0) / 2.0, -1.0, 1.0);
			return std::acos(cosine);
		}

		std::vector<double> cumulativeDistances(const PoseList& poses)
		{
			std::vector<double> distances(poses.size(), 0.0);
			for (std::size_t i = 1; i < poses.size(); ++i) {
				distances[i] = distances[i - 1] +
							   (poses[i].translation() - poses[i - 1].translation()).norm();
			}
			return distances;
		}

		// Top view (x right, z forward) of truth vs aligned estimate, plus ATE over frames.
		void drawPlot(const std::string& path, const PoseList& truth,
					  const std::vector<Eigen::Vector3d>& aligned,
					  const std::vector<double>& ateErrors, const EvaluationResult& result)
		{
			const int width = 1200;
			const int mapHeight = 900;
			const int chartHeight = 220;
			const int margin = 40;
			cv::Mat canvas(mapHeight + chartHeight, width, CV_8UC3, cv::Scalar(255, 255, 255));

			double minX = 1e18, maxX = -1e18, minZ = 1e18, maxZ = -1e18;
			auto expand = [&](double x, double z) {
				minX = std::min(minX, x); maxX = std::max(maxX, x);
				minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
			};
			for (const auto& pose : truth) expand(pose.translation().x(), pose.translation().z());
			for (const auto& p : aligned) expand(p.x(), p.z());

			const double scale = std::min((width - 2.0 * margin) / std::max(maxX - minX, 1.0),
										  (mapHeight - 2.0 * margin) / std::max(maxZ - minZ, 1.0));
			auto toPixel = [&](double x, double z) {
				return cv::Point(static_cast<int>(margin + (x - minX) * scale),
								 static_cast<int>(mapHeight - margin - (z - minZ) * scale));  // z up on screen
			};

			std::vector<cv::Point> truthLine, estimateLine;
			for (const auto& pose : truth) truthLine.push_back(toPixel(pose.translation().x(), pose.translation().z()));
			for (const auto& p : aligned) estimateLine.push_back(toPixel(p.x(), p.z()));

			cv::polylines(canvas, truthLine, false, cv::Scalar(70, 70, 70), 2, cv::LINE_AA);
			cv::polylines(canvas, estimateLine, false, cv::Scalar(32, 112, 224), 2, cv::LINE_AA);  // orange (BGR)
			cv::circle(canvas, truthLine.front(), 6, cv::Scalar(0, 160, 0), cv::FILLED, cv::LINE_AA);

			// Scale bar: 50 m.
			const int bar = static_cast<int>(50.0 * scale);
			cv::line(canvas, {width - margin - bar, mapHeight - 20}, {width - margin, mapHeight - 20},
					 cv::Scalar(0, 0, 0), 2);
			cv::putText(canvas, "50 m", {width - margin - bar, mapHeight - 28},
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

			std::ostringstream title;
			title << std::fixed << std::setprecision(2) << "gray = ground truth, orange = RigVis (aligned)   |   ATE "
				  << result.ate_m << " m";
			if (result.hasKittiMetric) {
				title << "   t_rel " << result.tRel_percent << " %   r_rel " << result.rRel_degPer100m << " deg/100m";
			}
			cv::putText(canvas, title.str(), {margin, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
						cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

			// ATE over frames.
			const int chartTop = mapHeight + 20;
			const int chartBottom = mapHeight + chartHeight - 30;
			const double maxError = std::max(*std::max_element(ateErrors.begin(), ateErrors.end()), 1e-6);
			cv::rectangle(canvas, {margin, chartTop}, {width - margin, chartBottom}, cv::Scalar(200, 200, 200), 1);
			std::vector<cv::Point> errorLine;
			for (std::size_t i = 0; i < ateErrors.size(); ++i) {
				const double fx = ateErrors.size() > 1 ? static_cast<double>(i) / static_cast<double>(ateErrors.size() - 1) : 0.0;
				errorLine.emplace_back(static_cast<int>(margin + fx * (width - 2 * margin)),
									   static_cast<int>(chartBottom - ateErrors[i] / maxError * (chartBottom - chartTop)));
			}
			cv::polylines(canvas, errorLine, false, cv::Scalar(32, 112, 224), 1, cv::LINE_AA);

			std::ostringstream chartLabel;
			chartLabel << std::fixed << std::setprecision(1) << "ATE per frame, max " << maxError
					   << " m   (frames " << result.firstFrame << " - " << result.firstFrame + result.frames - 1 << ")";
			cv::putText(canvas, chartLabel.str(), {margin, chartBottom + 22}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
						cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

			if (!cv::imwrite(path, canvas)) {
				std::cout << "Could not write plot to " << path << '\n';
			} else {
				std::cout << "Plot saved to " << path << '\n';
			}
		}
	}

	// ------------------------------------------------------------- file I/O

	PoseList loadKittiPoses(const std::string& path)
	{
		std::ifstream file(path);
		if (!file) throw std::runtime_error("Could not open trajectory: " + path);

		PoseList poses;
		std::string line;
		std::size_t lineNumber = 0;
		while (std::getline(file, line)) {
			++lineNumber;
			std::istringstream numbers(line);
			std::vector<double> values;
			double value = 0.0;
			while (numbers >> value) values.push_back(value);
			if (values.empty()) continue;
			if (values.size() != 12) {
				throw std::runtime_error(path + ":" + std::to_string(lineNumber) +
										 ": expected 12 numbers, got " + std::to_string(values.size()));
			}
			Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 4; ++column) {
					pose.matrix()(row, column) = values[static_cast<std::size_t>(row * 4 + column)];
				}
			}
			pose.linear() = orthonormalize(pose.linear());  // file has only 9 digits
			poses.push_back(pose);
		}
		return poses;
	}

	void writeKittiPose(std::ostream& out, const Eigen::Isometry3d& worldFromCamera)
	{
		const Eigen::Matrix4d& m = worldFromCamera.matrix();
		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 4; ++column) {
				out << m(row, column) << ((row == 2 && column == 3) ? '\n' : ' ');
			}
		}
	}

	// ---------------------------------------------------------- ground truth

	PoseList groundTruthFromOxts(const std::string& drivePath)
	{
		const fs::path drive = fs::absolute(drivePath);
		const fs::path day = drive.parent_path();
		const fs::path oxtsDirectory = drive / "oxts" / "data";

		std::vector<fs::path> oxtsFiles;
		for (const auto& entry : fs::directory_iterator(oxtsDirectory)) {
			if (entry.path().extension() == ".txt") oxtsFiles.push_back(entry.path());
		}
		std::sort(oxtsFiles.begin(), oxtsFiles.end());
		if (oxtsFiles.empty()) throw std::runtime_error("No OXTS files in " + oxtsDirectory.string());

		// T_cam_imu = R_rect_00 * T_cam0_velo * T_velo_imu
		const fs::path imuToVeloPath = day / "calib_imu_to_velo.txt";
		const fs::path veloToCamPath = day / "calib_velo_to_cam.txt";
		const fs::path camToCamPath = day / "calib_cam_to_cam.txt";
		const auto imuToVelo = readCalibrationFile(imuToVeloPath);
		const auto veloToCam = readCalibrationFile(veloToCamPath);
		const auto camToCam = readCalibrationFile(camToCamPath);

		const Eigen::Isometry3d veloFromImu = makeRigid(requireKey(imuToVelo, "R", 9, imuToVeloPath),
														requireKey(imuToVelo, "T", 3, imuToVeloPath));
		const Eigen::Isometry3d camFromVelo = makeRigid(requireKey(veloToCam, "R", 9, veloToCamPath),
														requireKey(veloToCam, "T", 3, veloToCamPath));
		const Eigen::Isometry3d rectification = makeRigid(requireKey(camToCam, "R_rect_00", 9, camToCamPath), {});
		const Eigen::Isometry3d imuFromCamera = (rectification * camFromVelo * veloFromImu).inverse();

		// KITTI raw devkit convertOxtsToPose: Mercator projection with the scale
		// of the first latitude; R = Rz(yaw) * Ry(pitch) * Rx(roll).
		constexpr double earthRadius = 6378137.0;
		double mercatorScale = 0.0;
		PoseList cameraPoses;
		cameraPoses.reserve(oxtsFiles.size());

		for (const fs::path& oxtsFile : oxtsFiles) {
			std::ifstream file(oxtsFile);
			double lat = 0, lon = 0, alt = 0, roll = 0, pitch = 0, yaw = 0;
			if (!(file >> lat >> lon >> alt >> roll >> pitch >> yaw)) {
				throw std::runtime_error("Could not parse OXTS file: " + oxtsFile.string());
			}
			if (cameraPoses.empty()) mercatorScale = std::cos(lat * kPi / 180.0);

			const double x = mercatorScale * lon * kPi * earthRadius / 180.0;
			const double y = mercatorScale * earthRadius * std::log(std::tan((90.0 + lat) * kPi / 360.0));

			Eigen::Isometry3d worldFromImu = Eigen::Isometry3d::Identity();
			worldFromImu.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
									 Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
									 Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
			worldFromImu.translation() = Eigen::Vector3d(x, y, alt);

			cameraPoses.push_back(worldFromImu * imuFromCamera);
		}

		const Eigen::Isometry3d firstInverse = cameraPoses.front().inverse();
		for (auto& pose : cameraPoses) pose = firstInverse * pose;
		return cameraPoses;
	}

	// ------------------------------------------------------------ evaluation

	EvaluationResult evaluateTrajectory(const PoseList& estimatedAll, const PoseList& groundTruthAll,
										const std::size_t firstFrame, const std::string& plotPath)
	{
		if (firstFrame >= groundTruthAll.size()) throw std::runtime_error("firstFrame beyond ground truth");
		const std::size_t count = std::min(estimatedAll.size(), groundTruthAll.size() - firstFrame);
		if (count < 2) throw std::runtime_error("Not enough overlapping frames to evaluate");

		// Paired frames, ground truth made relative to its first paired frame like the estimate.
		PoseList estimated(estimatedAll.begin(), estimatedAll.begin() + static_cast<long>(count));
		PoseList truth;
		truth.reserve(count);
		const Eigen::Isometry3d truthFirstInverse = groundTruthAll[firstFrame].inverse();
		for (std::size_t i = 0; i < count; ++i) truth.push_back(truthFirstInverse * groundTruthAll[firstFrame + i]);

		EvaluationResult result;
		result.frames = count;
		result.firstFrame = firstFrame;

		const std::vector<double> trueDistances = cumulativeDistances(truth);
		result.trueLength_m = trueDistances.back();
		result.estimatedLength_m = cumulativeDistances(estimated).back();

		// ---- ATE: best rigid alignment (Umeyama without scale), then RMSE.
		Eigen::Matrix3Xd estimatedPositions(3, static_cast<long>(count));
		Eigen::Matrix3Xd truePositions(3, static_cast<long>(count));
		for (std::size_t i = 0; i < count; ++i) {
			estimatedPositions.col(static_cast<long>(i)) = estimated[i].translation();
			truePositions.col(static_cast<long>(i)) = truth[i].translation();
		}
		const Eigen::Matrix4d alignment = Eigen::umeyama(estimatedPositions, truePositions, false);

		std::vector<Eigen::Vector3d> aligned(count);
		std::vector<double> ateErrors(count);
		double squaredSum = 0.0;
		for (std::size_t i = 0; i < count; ++i) {
			aligned[i] = alignment.topLeftCorner<3, 3>() * estimated[i].translation() + alignment.topRightCorner<3, 1>();
			ateErrors[i] = (aligned[i] - truth[i].translation()).norm();
			squaredSum += ateErrors[i] * ateErrors[i];
		}
		result.ate_m = std::sqrt(squaredSum / static_cast<double>(count));
		result.ateMax_m = *std::max_element(ateErrors.begin(), ateErrors.end());

		// ---- KITTI relative metric (devkit evaluate_odometry.cpp).
		double translationSum = 0.0;
		double rotationSum = 0.0;
		for (std::size_t first = 0; first < count; first += kStartFrameStep) {
			for (const double length : kSegmentLengths) {
				const auto reached = std::lower_bound(trueDistances.begin() + static_cast<long>(first),
													  trueDistances.end(), trueDistances[first] + length);
				if (reached == trueDistances.end()) continue;
				const std::size_t last = static_cast<std::size_t>(reached - trueDistances.begin());

				const Eigen::Isometry3d deltaTrue = truth[first].inverse() * truth[last];
				const Eigen::Isometry3d deltaEstimated = estimated[first].inverse() * estimated[last];
				const Eigen::Isometry3d error = deltaEstimated.inverse() * deltaTrue;

				translationSum += error.translation().norm() / length;
				rotationSum += rotationAngle(error.linear()) / length;
				++result.segmentCount;
			}
		}
		if (result.segmentCount > 0) {
			result.hasKittiMetric = true;
			result.tRel_percent = 100.0 * translationSum / static_cast<double>(result.segmentCount);
			result.rRel_degPer100m = 100.0 * kRadToDeg * rotationSum / static_cast<double>(result.segmentCount);
		}

		// ---- 1-frame relative error.
		double rpeTranslationSum = 0.0;
		double rpeRotationSum = 0.0;
		for (std::size_t i = 1; i < count; ++i) {
			const Eigen::Isometry3d deltaTrue = truth[i - 1].inverse() * truth[i];
			const Eigen::Isometry3d deltaEstimated = estimated[i - 1].inverse() * estimated[i];
			const Eigen::Isometry3d error = deltaEstimated.inverse() * deltaTrue;
			const double translation = error.translation().norm();
			const double rotation = rotationAngle(error.linear()) * kRadToDeg;
			rpeTranslationSum += translation;
			rpeRotationSum += rotation;
			if (translation > result.worstTranslation_m) {
				result.worstTranslation_m = translation;
				result.worstRotation_deg = rotation;
				result.worstFrame = firstFrame + i;
			}
		}
		result.rpeTranslationMean_m = rpeTranslationSum / static_cast<double>(count - 1);
		result.rpeRotationMean_deg = rpeRotationSum / static_cast<double>(count - 1);

		if (!plotPath.empty()) drawPlot(plotPath, truth, aligned, ateErrors, result);
		return result;
	}

	// ---------------------------------------------------------------- output

	void printEvaluation(const EvaluationResult& r)
	{
		std::ostringstream out;
		out << std::fixed;
		out << "\n================ RigVis vs KITTI ground truth ================\n";
		out << "Frames evaluated      : " << r.frames << "  (gt frames " << r.firstFrame << "-"
			<< r.firstFrame + r.frames - 1 << ")\n";
		out << std::setprecision(1) << "Path length           : truth " << std::setw(8) << r.trueLength_m
			<< " m | estimate " << std::setw(8) << r.estimatedLength_m << " m ("
			<< std::showpos << 100.0 * (r.estimatedLength_m / r.trueLength_m - 1.0) << std::noshowpos << " %)\n";
		out << std::setprecision(3) << "ATE (RMSE, aligned)   : " << std::setw(8) << r.ate_m << " m   (max "
			<< std::setprecision(2) << r.ateMax_m << " m)\n";
		if (r.hasKittiMetric) {
			out << std::setprecision(3) << "KITTI t_rel           : " << std::setw(8) << r.tRel_percent
				<< " %       over " << r.segmentCount << " segments\n";
			out << "KITTI r_rel           : " << std::setw(8) << r.rRel_degPer100m << " deg/100m\n";
		} else {
			out << "KITTI t_rel / r_rel   : n/a (needs >= 100 m of driving)\n";
		}
		out << std::setprecision(3) << "RPE per frame (mean)  : " << r.rpeTranslationMean_m << " m, "
			<< r.rpeRotationMean_deg << " deg\n";
		out << "Worst frame           : " << r.worstFrame << "  (" << r.worstTranslation_m << " m, "
			<< r.worstRotation_deg << " deg)\n";
		out << "Reference (seq 00)    : ORB-SLAM2 stereo ~0.70 %, ~0.25 deg/100m (with BA + loop closure)\n";
		out << "=============================================================\n";
		std::cout << out.str() << std::flush;
	}

	void appendEvaluationCsv(const std::string& csvPath, const EvaluationResult& r, const std::string& label)
	{
		const bool newFile = !fs::exists(csvPath);
		std::ofstream csv(csvPath, std::ios::app);
		if (!csv) {
			std::cout << "Could not open " << csvPath << " for writing\n";
			return;
		}
		if (newFile) {
			csv << "time,label,frames,first_frame,length_m,ate_m,t_rel_pct,r_rel_deg_per_100m,"
				   "rpe_t_mean_m,rpe_r_mean_deg,worst_frame\n";
		}

		const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		std::tm local{};
		localtime_r(&now, &local);

		csv << std::put_time(&local, "%Y-%m-%dT%H:%M:%S") << ',' << label << ',' << r.frames << ','
			<< r.firstFrame << ',' << std::fixed << std::setprecision(1) << r.trueLength_m << ','
			<< std::setprecision(3) << r.ate_m << ',';
		if (r.hasKittiMetric) csv << r.tRel_percent << ',' << r.rRel_degPer100m;
		else csv << ',';
		csv << ',' << std::setprecision(4) << r.rpeTranslationMean_m << ',' << r.rpeRotationMean_deg << ','
			<< r.worstFrame << '\n';
		std::cout << "Summary appended to " << csvPath << '\n';
	}
}
