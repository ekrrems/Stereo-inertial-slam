/* RigVis loop closure.

   After every new keyframe:
     1. Candidates   old keyframes near the current position estimate, looking in
                     a similar direction, not already covisible (proximity search,
                     as in LIO-SAM; works while drift < loopSearchRadiusMeters_).
     2. Verification match the current keyframe against the candidate's local map
                     (descriptors) and solve PnP RANSAC. Strict inlier thresholds:
                     a wrong loop would destroy the map.
     3. Consistency  correct only if the previous keyframe also verified a loop to
                     the same place.
     4. Correction   pose graph over all keyframes (PoseGraph.cpp), move every map
                     point with its anchor keyframe, merge the duplicate points,
                     then global bundle adjustment.
   DetectOnly mode runs 1-3 and logs, without changing the map.
*/
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#include "PoseGraph.hpp"
#include "StereoFrontend.hpp"

namespace RigVis
{
	namespace
	{
		constexpr double kRadToDeg = 57.29577951308232;

		double rotationAngleDegrees(const Eigen::Matrix3d& rotation)
		{
			const double cosine = std::clamp((rotation.trace() - 1.0) / 2.0, -1.0, 1.0);
			return std::acos(cosine) * kRadToDeg;
		}
	}

	void StereoFrontend::detectAndCloseLoop()
	{
		if (loopClosureMode_ == LoopClosureMode::Off || keyframes_.empty()) return;
		const std::size_t current = keyframes_.back().id;
		if (loopsClosed_ > 0 && current < lastLoopKeyframe_ + loopCooldownKeyframes_) return;

		LoopVerification best;
		for (const std::size_t candidate : findLoopCandidates()) {
			LoopVerification verification = verifyLoopCandidate(candidate);
			if (verification.accepted && verification.inliers > best.inliers) best = std::move(verification);
		}
		if (!best.accepted) {
			pendingLoopCandidate_ = -1;
			return;
		}

		const Eigen::Isometry3d& currentPose = keyframes_.back().worldFromCamera;
		const double meters = (best.worldFromCamera.translation() - currentPose.translation()).norm();
		const double degrees = rotationAngleDegrees(currentPose.linear().transpose() * best.worldFromCamera.linear());
		stats_.loopCandidate = static_cast<long>(best.candidateKeyframe);
		stats_.loopInliers = best.inliers;
		stats_.loopCorrectionMeters = meters;

		// Consistency: the previous keyframe (or the one before) must have
		// verified a loop to (almost) the same old keyframe.
		const bool consistent = pendingLoopCandidate_ >= 0 && current <= pendingLoopKeyframe_ + 2 &&
								std::abs(pendingLoopCandidate_ - static_cast<long>(best.candidateKeyframe)) <= 10;
		pendingLoopCandidate_ = static_cast<long>(best.candidateKeyframe);
		pendingLoopKeyframe_ = current;

		std::cout << std::fixed << std::setprecision(2) << "Loop verified: kf " << current << " <-> kf "
				  << best.candidateKeyframe << " | inliers " << best.inliers << " / " << best.matches
				  << " | drift " << meters << " m, " << degrees << " deg"
				  << (consistent ? "" : " | waiting for a consistent second detection") << '\n';

		if (!consistent || loopClosureMode_ == LoopClosureMode::DetectOnly) return;
		correctLoop(best);
	}

	std::vector<std::size_t> StereoFrontend::findLoopCandidates() const
	{
		const Keyframe& current = keyframes_.back();
		if (current.id < loopMinKeyframeGap_) return {};

		const Eigen::Vector3d position = current.worldFromCamera.translation();
		const Eigen::Vector3d forward = current.worldFromCamera.linear() * Eigen::Vector3d::UnitZ();
		const double minCosine = std::cos(loopMaxViewAngleDegrees_ / kRadToDeg);

		std::unordered_set<std::size_t> covisible;
		if (current.id < covisibilityWeights_.size()) {
			for (const auto& [id, weight] : covisibilityWeights_[current.id]) {
				if (weight >= covisibilityMinWeight_) covisible.insert(id);
			}
		}

		std::vector<std::pair<double, std::size_t>> nearby;  // (distance, keyframe)
		for (std::size_t i = 0; i + loopMinKeyframeGap_ <= current.id; ++i) {
			if (covisible.count(i)) continue;  // already connected: not a loop
			const Keyframe& keyframe = keyframes_[i];
			const double distance = (keyframe.worldFromCamera.translation() - position).norm();
			if (distance > loopSearchRadiusMeters_) continue;
			const Eigen::Vector3d otherForward = keyframe.worldFromCamera.linear() * Eigen::Vector3d::UnitZ();
			if (forward.dot(otherForward) < minCosine) continue;  // looking elsewhere: ORB won't match
			nearby.emplace_back(distance, i);
		}
		std::sort(nearby.begin(), nearby.end());

		// Closest few, spread out (not three neighbours of the same keyframe).
		std::vector<std::size_t> candidates;
		for (const auto& [distance, id] : nearby) {
			const bool tooClose = std::any_of(candidates.begin(), candidates.end(), [&](std::size_t c) {
				return (c > id ? c - id : id - c) < 5;
			});
			if (tooClose) continue;
			candidates.push_back(id);
			if (candidates.size() >= loopMaxCandidates_) break;
		}
		return candidates;
	}

	// Match the current keyframe (= current frame) against the map points of the
	// candidate and its covisible neighbours, then PnP RANSAC. Gives the current
	// pose expressed in the OLD part of the map.
	StereoFrontend::LoopVerification StereoFrontend::verifyLoopCandidate(const std::size_t candidateKeyframe) const
	{
		LoopVerification verification;
		verification.candidateKeyframe = candidateKeyframe;

		std::vector<std::size_t> keyframeIds{candidateKeyframe};
		for (const std::size_t id : bestCovisibleKeyframes(candidateKeyframe, 10, covisibilityMinWeight_)) {
			keyframeIds.push_back(id);
		}

		std::unordered_set<int> seen;
		std::vector<int> points;
		for (const std::size_t k : keyframeIds) {
			for (const int id : keyframes_.at(k).mapPointIds) {
				if (id < 0 || !mapPoints_.at(static_cast<std::size_t>(id)).valid) continue;
				if (seen.insert(id).second) points.push_back(id);
			}
		}
		if (points.size() < loopMinInliers_) return verification;

		const LocalMapMatches matches = matchLocalMapByDescriptor(points);
		verification.matches = matches.size();
		if (matches.size() < loopMinInliers_) return verification;

		cv::Mat rvec, tvec, inlierIndices;
		const bool solved = cv::solvePnPRansac(matches.objectPoints, matches.imagePoints, cameraMatrix_,
											   distortionCoefficients_, rvec, tvec, false, 300, 3.0, 0.999,
											   inlierIndices, cv::SOLVEPNP_ITERATIVE);
		if (!solved) return verification;
		verification.inliers = static_cast<std::size_t>(inlierIndices.rows);
		const double ratio = static_cast<double>(verification.inliers) / static_cast<double>(matches.size());
		if (verification.inliers < loopMinInliers_ || ratio < loopMinInlierRatio_) return verification;

		cv::Mat rotationMatrix;
		cv::Rodrigues(rvec, rotationMatrix);
		Eigen::Isometry3d cameraFromWorld = Eigen::Isometry3d::Identity();
		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 3; ++column) {
				cameraFromWorld.linear()(row, column) = rotationMatrix.at<double>(row, column);
			}
			cameraFromWorld.translation()(row) = tvec.at<double>(row, 0);
		}
		verification.worldFromCamera = cameraFromWorld.inverse();

		// Sanity: the correction cannot be larger than the search radius.
		const double jump =
			(verification.worldFromCamera.translation() - keyframes_.back().worldFromCamera.translation()).norm();
		if (jump > loopSearchRadiusMeters_) return verification;

		for (int row = 0; row < inlierIndices.rows; ++row) {
			const int i = inlierIndices.at<int>(row, 0);
			verification.inlierPairs.emplace_back(matches.keypointIndices[static_cast<std::size_t>(i)],
												  matches.mapPointIndices[static_cast<std::size_t>(i)]);
		}
		verification.accepted = true;
		return verification;
	}

	void StereoFrontend::correctLoop(const LoopVerification& loop)
	{
		const std::size_t n = keyframes_.size();
		const std::size_t current = n - 1;

		// ---- Pose graph. Edges from the poses BEFORE correction keep the local shape.
		std::vector<Eigen::Isometry3d> oldPoses(n);
		for (std::size_t i = 0; i < n; ++i) oldPoses[i] = keyframes_[i].worldFromCamera;
		std::vector<Eigen::Isometry3d> poses = oldPoses;

		std::vector<PoseGraphEdge> edges;
		auto addEdge = [&](std::size_t from, std::size_t to, const Eigen::Isometry3d& measured) {
			PoseGraphEdge edge;
			edge.from = from;
			edge.to = to;
			edge.measuredFromTo = measured;
			edges.push_back(edge);
		};
		for (std::size_t i = 1; i < n; ++i) addEdge(i - 1, i, oldPoses[i - 1].inverse() * oldPoses[i]);
		for (std::size_t i = 0; i < n && i < covisibilityWeights_.size(); ++i) {
			for (const auto& [j, weight] : covisibilityWeights_[i]) {
				if (j > i + 1 && j < n && weight >= 100) addEdge(i, j, oldPoses[i].inverse() * oldPoses[j]);
			}
		}
		addEdge(loop.candidateKeyframe, current, oldPoses[loop.candidateKeyframe].inverse() * loop.worldFromCamera);

		std::vector<bool> fixed(n, false);
		fixed[0] = true;  // world frame
		const PoseGraphSummary graph = optimizePoseGraph(poses, edges, fixed);
		if (!graph.ran) return;

		// ---- Move keyframes, and every point with its anchor keyframe.
		std::vector<Eigen::Isometry3d> correction(n);
		for (std::size_t i = 0; i < n; ++i) {
			correction[i] = poses[i] * oldPoses[i].inverse();
			keyframes_[i].worldFromCamera = poses[i];
		}
		for (SparseMapPoint& point : mapPoints_) {
			if (!point.valid) continue;
			const std::size_t anchor = point.observations.empty() ? point.firstKeyframeId
																  : point.observations.front().keyframeId;
			if (anchor < n) point.positionWorld = correction[anchor] * point.positionWorld;
		}
		worldFromCamera_ = keyframes_.back().worldFromCamera;

		// ---- Old and new map now overlap: merge the duplicate points.
		fuseLoopMapPoints(loop);
		updateCovisibility(current);

		std::cout << std::fixed << std::setprecision(1) << "Loop corrected: pose graph " << graph.nodes << " keyframes, "
				  << graph.edges << " edges, " << graph.iterations << " it, " << graph.milliseconds << " ms\n";

		// ---- Global bundle adjustment: the local BA with every keyframe free (except 0).
		if (globalBundleAdjustmentAfterLoop_) {
			BundleAdjustmentOptions options = localBundleAdjustmentOptions_;
			options.windowKeyframes = n;
			const BundleAdjustmentSummary global = runLocalBundleAdjustment(keyframes_, mapPoints_, calibration_, options);
			if (global.ran) {
				worldFromCamera_ = keyframes_.back().worldFromCamera;
				std::cout << std::fixed << std::setprecision(2) << "Global BA: " << global.points << " points, "
						  << global.observations << " obs | rms " << global.rmsNormalizedBefore << " -> "
						  << global.rmsNormalizedAfter << " sigma | outliers " << global.outliersRemoved << " | "
						  << std::setprecision(0) << global.milliseconds << " ms\n";
			}
		}

		stats_.loopCorrected = true;
		++loopsClosed_;
		lastLoopKeyframe_ = current;
		pendingLoopCandidate_ = -1;
	}

	// Every verified (current keypoint, old point) pair: the keypoint should
	// observe the OLD point. If it already has a new duplicate point, merge the
	// duplicate's observations into the old one and retire the duplicate.
	void StereoFrontend::fuseLoopMapPoints(const LoopVerification& loop)
	{
		Keyframe& currentKeyframe = keyframes_.back();
		std::vector<double> rightXByLeftIndex(currentLeft_.keypoints.size(), -1.0);
		for (const StereoMatch& match : currentStereoMatches_) {
			rightXByLeftIndex.at(static_cast<std::size_t>(match.leftKeypointIndex)) =
				currentRight_.keypoints.at(static_cast<std::size_t>(match.rightKeypointIndex)).pt.x;
		}

		auto observes = [](const SparseMapPoint& point, std::size_t keyframeId) {
			return std::any_of(point.observations.begin(), point.observations.end(),
							   [&](const Observation& o) { return o.keyframeId == keyframeId; });
		};

		std::size_t fused = 0;
		for (const auto& [keypoint, oldIndex] : loop.inlierPairs) {
			const std::size_t k = static_cast<std::size_t>(keypoint);
			SparseMapPoint& oldPoint = mapPoints_.at(static_cast<std::size_t>(oldIndex));
			const int newIndex = currentKeyframe.mapPointIds.at(k);
			if (newIndex == oldIndex || !oldPoint.valid) continue;

			if (newIndex >= 0) {
				SparseMapPoint& duplicate = mapPoints_.at(static_cast<std::size_t>(newIndex));
				for (const Observation& observation : duplicate.observations) {
					std::vector<int>& ids = keyframes_.at(observation.keyframeId).mapPointIds;
					if (observes(oldPoint, observation.keyframeId)) {
						ids.at(static_cast<std::size_t>(observation.keypointIndex)) = -1;  // keyframe already has it
					} else {
						oldPoint.observations.push_back(observation);
						ids.at(static_cast<std::size_t>(observation.keypointIndex)) = oldIndex;
					}
				}
				duplicate.observations.clear();
				duplicate.valid = false;
			} else if (!observes(oldPoint, currentKeyframe.id)) {
				Observation observation;
				observation.keyframeId = currentKeyframe.id;
				observation.keypointIndex = keypoint;
				observation.pixel = Eigen::Vector2d(currentKeyframe.keypoints[k].pt.x, currentKeyframe.keypoints[k].pt.y);
				observation.uRight = rightXByLeftIndex[k];
				observation.octave = currentKeyframe.keypoints[k].octave;
				oldPoint.observations.push_back(observation);
				currentKeyframe.mapPointIds[k] = oldIndex;
			} else {
				continue;
			}
			currentLeft_.mapPointIds.at(k) = currentKeyframe.mapPointIds[k];
			++fused;
		}
		std::cout << "Loop fusion: " << fused << " current keypoints now observe old map points\n";
	}
}
