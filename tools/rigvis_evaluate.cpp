/*
	Evaluate a saved RigVis trajectory against KITTI ground truth, without rerunning.

	Usage:
	  ./rigvis_evaluate <trajectory_kitti.txt> <..._drive_XXXX_sync folder> [first_frame] [plot.png]

	first_frame: ground-truth frame that line 0 of the trajectory belongs to (default 0).
*/
#include <exception>
#include <iostream>
#include <string>

#include "Evaluation.hpp"

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0]
				  << " <trajectory_kitti.txt> <drive_sync_folder> [first_frame] [plot.png]\n";
		return 1;
	}

	const std::string trajectoryPath = argv[1];
	const std::string drivePath = argv[2];
	const std::size_t firstFrame = argc > 3 ? std::stoul(argv[3]) : 0;
	const std::string plotPath = argc > 4 ? argv[4] : "trajectory_eval.png";

	try {
		const RigVis::PoseList estimated = RigVis::loadKittiPoses(trajectoryPath);
		const RigVis::PoseList groundTruth = RigVis::groundTruthFromOxts(drivePath);
		const RigVis::EvaluationResult result =
			RigVis::evaluateTrajectory(estimated, groundTruth, firstFrame, plotPath);
		RigVis::printEvaluation(result);
		RigVis::appendEvaluationCsv("evaluation_results.csv", result, trajectoryPath);
	} catch (const std::exception& error) {
		std::cerr << "Evaluation failed: " << error.what() << '\n';
		return 1;
	}
	return 0;
}
