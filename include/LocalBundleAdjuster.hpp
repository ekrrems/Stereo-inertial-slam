/* RigVis local bundle adjustment (Ceres).

   Adjusts the last N keyframe poses and the map points they observe so that
   all observations reproject as well as possible:

       minimise  sum_ij  rho( || (z_ij - pi(T_i, P_j)) / sigma_ij ||^2 )

   z_ij  = measured (u_L, v, u_R) or (u_L, v)        -> the observation vector l
   pi    = rectified stereo projection                -> f(x)
   sigma = 1.2^octave px                              -> weights P = 1/sigma^2
   rho   = Huber, threshold from chi^2 (95 %)         -> robust estimation

   Keyframes outside the window that also see those points stay FIXED: their
   observations still count, their poses do not move (they fix the datum).
   The function only reads/writes the frontend's own structs, so the solver
   behind it can be swapped without touching the frontend.
*/
#pragma once

#include <cstddef>
#include <vector>

#include "types.hpp"

namespace RigVis
{
	struct Keyframe;
	struct SparseMapPoint;

	struct BundleAdjustmentOptions
	{
		std::size_t windowKeyframes = 10;       // free keyframes (newest)
		std::size_t minObservationsPerPoint = 2; // points with 1 observation add no redundancy
		int iterationsFirstPass = 5;            // before outlier removal
		int iterationsSecondPass = 10;          // after outlier removal
		double chi2Stereo = 7.815;              // chi^2, 3 DoF, 95 %
		double chi2Mono = 5.991;                // chi^2, 2 DoF, 95 %
		int numThreads = 4;
	};

	struct BundleAdjustmentSummary
	{
		bool ran = false;
		std::size_t freeKeyframes = 0;
		std::size_t fixedKeyframes = 0;
		std::size_t points = 0;
		std::size_t observations = 0;
		std::size_t outliersRemoved = 0;

		// Ceres cost = 0.5 * sum of (robustified) squared normalized residuals.
		double initialCost = 0.0;
		double finalCost = 0.0;
		int iterations = 0;
		double milliseconds = 0.0;

		// RMS of the normalized residuals (in sigma units, ~1 means the weights
		// match the noise: the a-posteriori variance factor check).
		double rmsNormalizedBefore = 0.0;
		double rmsNormalizedAfter = 0.0;
	};

	// Optimises keyframes[size-window .. size-1] and their points in place.
	// Keyframe 0 is never free (it fixes the world frame). Outlier observations
	// are removed from the map point AND from the keyframe's mapPointIds, so
	// tracking stops using them too.
	BundleAdjustmentSummary runLocalBundleAdjustment(std::vector<Keyframe>& keyframes,
													 std::vector<SparseMapPoint>& mapPoints,
													 const StereoCalibration& calibration,
													 const BundleAdjustmentOptions& options);
}
