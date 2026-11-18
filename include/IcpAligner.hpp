/* RigVis point-to-plane ICP between two dense keyframe clouds.

   The dense SGBM clouds (DenseMapper) of two neighbouring keyframes overlap.
   ICP finds the rigid motion that lays the SOURCE cloud (keyframe B) onto the
   TARGET cloud (keyframe A), starting from the motion the tracker estimated:

       minimise  sum_i  w_i * ( n_i . (T * p_i - q_i) )^2

   p_i  source point,  q_i  nearest target point,  n_i  target surface normal.
   Point-to-plane lets points slide along a surface (a wall, the road), so the
   result is driven by the geometry, not by the exact sampling of the two clouds.

   The result is a RELATIVE POSE MEASUREMENT between the two keyframes plus its
   information matrix (6x6). Directions the scene does not constrain (e.g. along
   a straight corridor) get a small information value, so bundle adjustment
   trusts ICP only where the geometry is informative.

   Runs on its own thread (std::async in StereoFrontend): it only reads copies
   of the two clouds, no shared state.
*/
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace RigVis
{
	struct IcpOptions
	{
		double voxelSize = 0.15;               // downsample both clouds (m)
		double maxRange = 15.0;                // m; stereo depth noise grows with Z^2, far points only add noise
		double normalRadius = 0.50;            // neighbourhood for the target normals (m)
		std::size_t minNormalNeighbours = 6;
		double maxPlanarity = 0.25;            // smallest/middle eigenvalue: only flat neighbourhoods get a normal
		int maxIterations = 25;
		double maxCorrespondenceStart = 1.00;  // m, shrinks linearly ...
		double maxCorrespondenceEnd = 0.30;    // ... to this over the iterations
		double huberDelta = 0.10;              // m, robust weights on the point-to-plane distance
		// Stereo depth noise grows with Z^2, so each correspondence is weighted by
		// 1 / (1 + (Z / depthWeightScale)^4) with Z the source point's depth in its own camera.
		double depthWeightScale = 6.0;         // m (0 = no depth weighting)
		double convergenceStep = 1e-5;         // stop when the update is this small

		// Acceptance tests (a bad ICP edge would pull BA the wrong way).
		std::size_t minInliers = 300;
		double minOverlap = 0.25;              // inliers / source points
		double maxRmse = 0.12;                 // m
		double maxCorrectionMeters = 0.50;     // ICP may refine the tracker's guess, not replace it
		double maxCorrectionDegrees = 3.0;
		double minConditionRatio = 1e-4;       // smallest / largest eigenvalue of the normalized information

		// ICP residuals of neighbouring points are strongly correlated, so H / rmse^2
		// is far too confident. Scale it as if only this many points were independent.
		// Calibrated on KITTI 00 against ground truth: SGBM-cloud ICP between keyframes
		// 3 frames apart is accurate to ~10 cm / 0.4 deg, which this value reproduces.
		double effectiveIndependentPoints = 1.5;
	};

	struct IcpResult
	{
		bool valid = false;
		std::string reason;                    // why it was rejected (empty if valid)
		Eigen::Isometry3d targetFromSource = Eigen::Isometry3d::Identity();
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();  // [rotation; translation]
		std::size_t sourcePoints = 0;
		std::size_t targetPoints = 0;
		std::size_t inliers = 0;
		double overlap = 0.0;
		double rmse = 0.0;                     // point-to-plane, m
		double correctionMeters = 0.0;
		double correctionDegrees = 0.0;
		int iterations = 0;
		double milliseconds = 0.0;
	};

	// target/source: points in their own keyframe camera frames.
	// initialTargetFromSource: T_A<-B from the current keyframe poses.
	IcpResult alignPointToPlane(const std::vector<Eigen::Vector3f>& target, const std::vector<Eigen::Vector3f>& source,
								const Eigen::Isometry3d& initialTargetFromSource, const IcpOptions& options = IcpOptions());
}
