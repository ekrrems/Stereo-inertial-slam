/* RigVis pose graph optimisation (Ceres).

   Nodes:  keyframe poses T_wc (world from camera), 6 DoF each. No map points.
   Edges:  relative pose measurements T_ij = T_i^-1 * T_j between two keyframes:
             - consecutive keyframes and strong covisibility links, taken from the
               poses BEFORE the loop correction (they keep the local shape), and
             - the loop edge, measured by geometric verification.
   Residual per edge (6-vector), as in Ceres' pose_graph_3d example:
             r = [ t_ij,est - t_ij,meas ;  2 * vec( q_ij,meas * q_ij,est^-1 ) ] / sigma
   Before the correction every odometry edge fits and the loop edge does not;
   least squares spreads that closing error over the whole chain (like
   distributing the misclosure of a traverse in surveying).
*/
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Geometry>

namespace RigVis
{
	struct PoseGraphEdge
	{
		std::size_t from = 0;
		std::size_t to = 0;
		Eigen::Isometry3d measuredFromTo = Eigen::Isometry3d::Identity();  // T_from^-1 * T_to
		double translationSigma = 0.1;   // m
		double rotationSigma = 0.01;     // rad
	};

	struct PoseGraphSummary
	{
		bool ran = false;
		std::size_t nodes = 0;
		std::size_t edges = 0;
		int iterations = 0;
		double initialCost = 0.0;
		double finalCost = 0.0;
		double milliseconds = 0.0;
	};

	// Optimises worldFromCamera in place. Nodes with fixed[i] == true do not move
	// (at least one must be fixed: it defines the world frame).
	PoseGraphSummary optimizePoseGraph(std::vector<Eigen::Isometry3d>& worldFromCamera,
									   const std::vector<PoseGraphEdge>& edges,
									   const std::vector<bool>& fixed,
									   int maxIterations = 50);
}
