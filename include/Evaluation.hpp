/* RigVis trajectory evaluation against KITTI ground truth (pure C++: Eigen + OpenCV). */
#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace RigVis
{
	// Camera-to-world poses of the left rectified camera, one per frame.
	using PoseList = std::vector<Eigen::Isometry3d>;

	struct EvaluationResult
	{
		std::size_t frames = 0;
		std::size_t firstFrame = 0;          // ground-truth frame paired with estimate frame 0

		double trueLength_m = 0.0;
		double estimatedLength_m = 0.0;

		// Absolute trajectory error: RMSE of positions after the best rigid
		// alignment (rotation + translation, no scale since stereo is metric).
		double ate_m = 0.0;
		double ateMax_m = 0.0;

		// Official KITTI odometry metric over 100..800 m segments.
		bool hasKittiMetric = false;          // false if less than 100 m was driven
		std::size_t segmentCount = 0;
		double tRel_percent = 0.0;
		double rRel_degPer100m = 0.0;

		// 1-frame relative pose error: shows where it goes wrong.
		double rpeTranslationMean_m = 0.0;
		double rpeRotationMean_deg = 0.0;
		std::size_t worstFrame = 0;           // ground-truth frame index
		double worstTranslation_m = 0.0;
		double worstRotation_deg = 0.0;
	};

	// KITTI odometry format: 12 numbers of the 3x4 [R | t] per line, row-major.
	PoseList loadKittiPoses(const std::string& path);
	void writeKittiPose(std::ostream& out, const Eigen::Isometry3d& worldFromCamera);

	// Ground truth from a raw KITTI *_sync drive: OXTS (GPS/INS) + calibration files
	// in the parent day folder. Same as the KITTI raw devkit's convertOxtsToPose,
	// transformed into the left rectified camera and made relative to frame 0.
	PoseList groundTruthFromOxts(const std::string& drivePath);

	// Pairs estimated[i] with groundTruth[firstFrame + i]. A run stopped early
	// just covers fewer frames. If plotPath is not empty, a PNG is written.
	EvaluationResult evaluateTrajectory(const PoseList& estimated,
										const PoseList& groundTruth,
										std::size_t firstFrame = 0,
										const std::string& plotPath = "");

	void printEvaluation(const EvaluationResult& result);

	// Appends one summary row per run (header written on first use).
	void appendEvaluationCsv(const std::string& csvPath,
							 const EvaluationResult& result,
							 const std::string& label);
}
