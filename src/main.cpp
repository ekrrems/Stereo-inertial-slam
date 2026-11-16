#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <csignal>
#include <exception>

#include "DataLoader.hpp"
#include "Estimator.hpp"
#include "Evaluation.hpp"
#include "TrajectoryViewer.hpp"
#include "StereoFrontend.hpp"
#include "MapViewer.hpp"

#include "types.hpp"

#include <opencv2/core.hpp>  // cv::FileStorage (YAML config)

// Set by CMake to <project>/config/rigvis.yaml. That file holds machine-specific
// paths and is git-ignored; copy config/rigvis.example.yaml to create it.
#ifndef RIGVIS_DEFAULT_CONFIG
#define RIGVIS_DEFAULT_CONFIG "config/rigvis.yaml"
#endif

namespace
{
	// First Ctrl+C: finish the current frame, leave the loop, still evaluate.
	// Second Ctrl+C: default behaviour, quit immediately.
	volatile std::sig_atomic_t stopRequested = 0;

	void onInterrupt(int)
	{
		stopRequested = 1;
		std::signal(SIGINT, SIG_DFL);
	}
}

int main(int argc, char** argv)
{
	std::signal(SIGINT, onInterrupt);

	// ---------- Config: paths live in a git-ignored YAML file, not in the code ----------
	// Usage: ./rigvis_dataset_runner [config.yaml]
	const std::string configPath = argc > 1 ? argv[1] : RIGVIS_DEFAULT_CONFIG;
	cv::FileStorage config(configPath, cv::FileStorage::READ);
	if (!config.isOpened()) {
		std::cerr << "Could not open config: " << configPath << "\n"
				  << "Copy config/rigvis.example.yaml to config/rigvis.yaml and set your dataset path.\n";
		return 1;
	}
	std::string dataset_path, imu_path;
	config["dataset_path"] >> dataset_path;
	config["imu_path"] >> imu_path;  // optional
	if (dataset_path.empty()) {
		std::cerr << "dataset_path is missing in " << configPath << '\n';
		return 1;
	}
	std::cout << "Config: " << configPath << "\n  dataset: " << dataset_path
			  << "\n  imu:     " << (imu_path.empty() ? "(10 Hz OXTS from the dataset)" : imu_path) << '\n';

	RigVis::DataLoader loader(dataset_path, imu_path);

	RigVis::SensorEvent event;

	RigVis::Estimator estimator{};

	// RigVis::TrajectoryViewer viewer{};
	RigVis::StereoFrontend frontend{loader.calibration()};
	RigVis::MapViewer mapViewer;
	std::size_t imuCount = 0;

	// Per-frame diagnostics. Written next to where the binary is run from.
	const std::string statsPath = "frame_stats.csv";
	std::ofstream statsCsv(statsPath);
	statsCsv << RigVis::frameStatsCsvHeader() << ",yaw_rate_dps\n";
	std::cout << "Writing per-frame stats to " << statsPath << '\n';

	// OXTS angular rate is in the vehicle frame (forward, left, up),
	// so z is the yaw rate. + = turning left.
	constexpr double radToDeg = 57.29577951308232;
	double latestYawRateDps = 0.0;

	// Estimated trajectory, one pose per stereo frame, compared with ground truth at the end.
	const std::string trajectoryPath = "trajectory_kitti.txt";
	std::ofstream trajectoryFile(trajectoryPath);
	trajectoryFile.precision(9);
	RigVis::PoseList estimatedTrajectory;

	while (!stopRequested && loader.next(event)) {
		if (std::holds_alternative<RigVis::ImuSample>(event)) {
			const auto& imu = std::get<RigVis::ImuSample>(event);
			estimator.imuPropagate(imu);
			latestYawRateDps = imu.angularVel.z() * radToDeg;

			// if (++imuCount % 5 == 0) {
			// 	viewer.update(
			// 		estimator.state(),
			// 		estimator.covariance());
			// }

			// std::cout << "IMU: " << imu.timestamp_s << std::endl;
		} else {
			const auto& stereo = std::get<RigVis::StereoFrame>(event);
			frontend.process(stereo);

			statsCsv << RigVis::frameStatsCsvRow(frontend.lastStats())
					 << ',' << latestYawRateDps << '\n';
			statsCsv.flush();  // keep the file readable even if you Ctrl+C mid-run

			// Written for every frame (also lost ones) so line i == frame i.
			RigVis::writeKittiPose(trajectoryFile, frontend.currentPose());
			estimatedTrajectory.push_back(frontend.currentPose());

			mapViewer.update(
				stereo,
				frontend.currentLeftFeatures(),
				frontend.currentRightFeatures(),
				frontend.mapPoints(),
				frontend.keyframes(),
				frontend.currentPose());

			// std::cout << "Stereo: " << stereo.timestamp_s
			// 		<< " | "
			// 		<< stereo.left.image.cols
			// 		<< " x "
			// 		<< stereo.left.image.rows
			// 		<< std::endl;
		}
	}

	// viewer.waitUntilClosed();

	// ---------- Compare with KITTI ground truth ----------
	statsCsv.close();
	trajectoryFile.close();

	if (stopRequested) {
		std::cout << "\nStopped early after " << estimatedTrajectory.size() << " frames; evaluating those.\n";
	}

	// Final trajectory: every frame rebuilt from the corrected keyframe poses, so
	// loop closures (and BA) also fix frames processed before them.
	// trajectory_kitti.txt keeps the online poses, for comparison.
	const RigVis::PoseList finalTrajectory = frontend.finalTrajectory();
	const std::string finalTrajectoryPath = "trajectory_final.txt";
	{
		std::ofstream finalFile(finalTrajectoryPath);
		finalFile.precision(9);
		for (const auto& pose : finalTrajectory) RigVis::writeKittiPose(finalFile, pose);
	}
	std::cout << "Loops closed: " << frontend.loopsClosed() << " | final trajectory saved to "
			  << finalTrajectoryPath << '\n';

	// 3D map files: open the .ply files in Metashape (File > Import > Import Points),
	// CloudCompare, MeshLab or Blender. z-up: x right, y forward, z up.
	frontend.exportSparseMapPly("map_sparse.ply");
	frontend.exportDenseMapPly("map_dense.ply", 0.10);  // 10 cm voxels
	frontend.exportCamerasPly("map_cameras.ply");
	frontend.exportKeyframesCsv("keyframes.csv");

	// Pure C++ (Evaluation.cpp). If it fails, the run's outputs are still on disk
	// and ./rigvis_evaluate trajectory_final.txt <drive> re-evaluates without rerunning.
	try {
		const RigVis::PoseList groundTruth = RigVis::groundTruthFromOxts(dataset_path);
		const RigVis::EvaluationResult result =
			RigVis::evaluateTrajectory(finalTrajectory, groundTruth, 0, "trajectory_eval.png");
		RigVis::printEvaluation(result);
		RigVis::appendEvaluationCsv("evaluation_results.csv", result,
									std::to_string(finalTrajectory.size()) + " frames / " +
										std::to_string(frontend.loopsClosed()) + " loops");  // no commas: CSV field
	} catch (const std::exception& error) {
		std::cout << "Evaluation failed: " << error.what() << '\n'
				  << "Trajectory saved to " << trajectoryPath << "; rerun with ./rigvis_evaluate.\n";
	}

	return 0;
}