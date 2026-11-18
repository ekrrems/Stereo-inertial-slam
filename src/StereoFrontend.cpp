#include "StereoFrontend.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgcodecs.hpp>  // imwrite (BA debug images)
#include <opencv2/imgproc.hpp>   // convexHull, contourArea, drawing

namespace RigVis
{
	// ------------------------------------------------------------ diagnostics

	std::string frameStatsCsvHeader()
	{
		return "frame,timestamp_s,features,stereo_matches,"
			   "local_map_points,local_map_visible,local_pairs,local_inliers,local_inlier_ratio,"
			   "fallback_pairs,fallback_inliers,tracked_map_points,grid_coverage,hull_area,"
			   "path,keyframe,kf_reason,"
			   "ba_ran,ba_ms,ba_outliers,ba_rms_before,ba_rms_after,"
			   "match_method,local_keyframes,associated_after_pose,culled_points,"
			   "loop_candidate,loop_inliers,loop_corrected,loop_correction_m,dense_ms,icp_edges";
	}

	std::string frameStatsCsvRow(const FrameStats& s)
	{
		const double localRatio = s.localPairs > 0
			? static_cast<double>(s.localInliers) / static_cast<double>(s.localPairs)
			: 0.0;

		std::ostringstream row;
		row << s.frameIndex << ',' << std::fixed << std::setprecision(6) << s.timestamp_s << ','
			<< s.features << ',' << s.stereoMatches << ','
			<< s.localMapPoints << ',' << s.localMapVisible << ','
			<< s.localPairs << ',' << s.localInliers << ','
			<< std::setprecision(3) << localRatio << ','
			<< s.fallbackPairs << ',' << s.fallbackInliers << ','
			<< s.trackedMapPoints << ',' << s.gridCoverage << ',' << s.hullAreaRatio << ','
			<< s.path << ',' << (s.keyframeInserted ? 1 : 0) << ',' << s.keyframeReason << ','
			<< (s.bundleAdjustmentRan ? 1 : 0) << ',' << std::setprecision(1) << s.bundleAdjustmentMilliseconds << ','
			<< s.bundleAdjustmentOutliers << ',' << std::setprecision(3) << s.bundleAdjustmentRmsBefore << ','
			<< s.bundleAdjustmentRmsAfter << ',' << s.matchMethod << ',' << s.localKeyframes << ','
			<< s.associatedAfterPose << ',' << s.culledPoints << ','
			<< s.loopCandidate << ',' << s.loopInliers << ',' << (s.loopCorrected ? 1 : 0) << ',' << s.loopCorrectionMeters
			<< ',' << std::setprecision(1) << s.denseMilliseconds << ',' << s.icpEdgesInBundleAdjustment;
		return row.str();
	}

	// Projects map points with a given pose and counts those inside the image
	// in front of the camera. Same test as ORB-SLAM's isInFrustum, minus the
	// viewing-angle and scale checks.
	std::size_t StereoFrontend::countVisibleMapPoints(
		const std::vector<int>& mapPointIndices,
		const Eigen::Isometry3d& worldFromCamera) const
	{
		if (imageWidth_ <= 0 || imageHeight_ <= 0) return 0;

		const Eigen::Isometry3d cameraFromWorld = worldFromCamera.inverse();
		std::size_t visible = 0;

		for (const int mapPointIndex : mapPointIndices) {
			const Eigen::Vector3d pointCamera =
				cameraFromWorld * mapPoints_.at(mapPointIndex).positionWorld;
			if (pointCamera.z() <= 0.0) continue;  // behind the camera

			const double u = calibration_.left.fx * pointCamera.x() / pointCamera.z() + calibration_.left.cx;
			const double v = calibration_.left.fy * pointCamera.y() / pointCamera.z() + calibration_.left.cy;
			if (u < 0.0 || u >= imageWidth_ || v < 0.0 || v >= imageHeight_) continue;  // out of view

			++visible;
		}
		return visible;
	}

	// Fraction of gridRows_ x gridCols_ cells that contain at least one pixel.
	// KITTI's top rows are mostly sky, so ~0.7-0.8 is already a good spread.
	double StereoFrontend::computeGridCoverage(const std::vector<cv::Point2f>& pixels) const
	{
		if (imageWidth_ <= 0 || imageHeight_ <= 0 || pixels.empty()) return 0.0;

		std::vector<bool> occupied(static_cast<std::size_t>(gridRows_ * gridCols_), false);
		for (const cv::Point2f& p : pixels) {
			const int col = std::clamp(static_cast<int>(p.x * gridCols_ / imageWidth_), 0, gridCols_ - 1);
			const int row = std::clamp(static_cast<int>(p.y * gridRows_ / imageHeight_), 0, gridRows_ - 1);
			occupied[static_cast<std::size_t>(row * gridCols_ + col)] = true;
		}
		const auto filled = std::count(occupied.begin(), occupied.end(), true);
		return static_cast<double>(filled) / static_cast<double>(occupied.size());
	}

	// Area enclosed by the outermost inliers, as a fraction of the image.
	// Unlike grid coverage it separates "30 cells clustered in the middle"
	// (small hull) from "30 cells spread to the edges" (large hull).
	// A single far outlier can inflate it, so read it together with coverage.
	double StereoFrontend::computeHullAreaRatio(const std::vector<cv::Point2f>& pixels) const
	{
		if (imageWidth_ <= 0 || imageHeight_ <= 0 || pixels.size() < 3) return 0.0;

		std::vector<cv::Point2f> hull;
		cv::convexHull(pixels, hull);
		const double hullArea = cv::contourArea(hull);
		return hullArea / (static_cast<double>(imageWidth_) * static_cast<double>(imageHeight_));
	}

	// ------------------------------------------------- reprojection report

	namespace
	{
		ReprojectionGroupStats summarize(std::vector<double> errors, const std::size_t chi2Outliers)
		{
			ReprojectionGroupStats stats;
			stats.count = errors.size();
			if (errors.empty()) return stats;

			std::sort(errors.begin(), errors.end());
			stats.medianPixels = errors[errors.size() / 2];
			stats.p90Pixels = errors[std::min(errors.size() - 1, errors.size() * 9 / 10)];
			const auto above5 = std::count_if(errors.begin(), errors.end(),
											  [](const double e) { return e > 5.0; });
			stats.fractionAbove5Pixels = static_cast<double>(above5) / static_cast<double>(errors.size());
			stats.fractionChi2Outliers = static_cast<double>(chi2Outliers) / static_cast<double>(errors.size());
			return stats;
		}
	}

	// For every observation: project the point with the keyframe's current pose
	// and compare with the measured pixels. This is Delta l = l - f(x0), the
	// input of bundle adjustment, computed with the same model the BA residual uses:
	//   P_c = T_cw * P_w,  u_L = fx X/Z + cx,  v = fy Y/Z + cy,  u_R = u_L - fx b / Z
	ReprojectionReport StereoFrontend::reprojectionReport(const std::size_t windowKeyframes) const
	{
		ReprojectionReport report;
		if (keyframes_.empty()) return report;

		const std::size_t firstKeyframe = (windowKeyframes == 0 || windowKeyframes >= keyframes_.size())
			? 0 : keyframes_.size() - windowKeyframes;
		report.keyframesChecked = keyframes_.size() - firstKeyframe;

		// T_cw for every checked keyframe, computed once. The stored pose is
		// worldFromCamera (T_wc); projection needs cameraFromWorld (T_cw).
		std::vector<Eigen::Isometry3d> cameraFromWorld(keyframes_.size());
		for (std::size_t k = firstKeyframe; k < keyframes_.size(); ++k) {
			cameraFromWorld[k] = keyframes_[k].worldFromCamera.inverse();
		}

		const double fx = calibration_.left.fx;
		const double fy = calibration_.left.fy;
		const double cx = calibration_.left.cx;
		const double cy = calibration_.left.cy;
		const double focalTimesBaseline = fx * calibration_.baseline_m;

		std::vector<double> anchorErrors, reobservedErrors;
		std::size_t anchorChi2 = 0, reobservedChi2 = 0;

		for (const SparseMapPoint& point : mapPoints_) {
			if (!point.valid) continue;

			for (std::size_t i = 0; i < point.observations.size(); ++i) {
				const Observation& observation = point.observations[i];
				if (observation.keyframeId < firstKeyframe || observation.keyframeId >= keyframes_.size()) continue;

				const Eigen::Vector3d pointCamera = cameraFromWorld[observation.keyframeId] * point.positionWorld;
				if (pointCamera.z() <= 0.0) {
					++report.behindCamera;
					continue;
				}

				const double inverseDepth = 1.0 / pointCamera.z();
				const double uLeft = fx * pointCamera.x() * inverseDepth + cx;
				const double v = fy * pointCamera.y() * inverseDepth + cy;

				// Residual vector v = observed - predicted (2 or 3 rows).
				double squaredPixels = (observation.pixel.x() - uLeft) * (observation.pixel.x() - uLeft) +
									   (observation.pixel.y() - v) * (observation.pixel.y() - v);
				double chi2Threshold = 5.991;  // chi^2, 2 DoF, 95 %
				if (observation.isStereo()) {
					const double uRight = uLeft - focalTimesBaseline * inverseDepth;
					squaredPixels += (observation.uRight - uRight) * (observation.uRight - uRight);
					chi2Threshold = 7.815;     // chi^2, 3 DoF, 95 %
					++report.stereoObservations;
				} else {
					++report.monoObservations;
				}

				// Weighted squared residual v'Pv with P = I / sigma^2, sigma = 1.2^octave.
				const double sigma = std::pow(1.2, observation.octave);
				const bool chi2Outlier = squaredPixels / (sigma * sigma) > chi2Threshold;

				// The first observation is made by the keyframe that created the point.
				if (i == 0) {
					anchorErrors.push_back(std::sqrt(squaredPixels));
					anchorChi2 += chi2Outlier ? 1 : 0;
				} else {
					reobservedErrors.push_back(std::sqrt(squaredPixels));
					reobservedChi2 += chi2Outlier ? 1 : 0;
				}
			}
		}

		report.anchor = summarize(std::move(anchorErrors), anchorChi2);
		report.reobserved = summarize(std::move(reobservedErrors), reobservedChi2);
		return report;
	}

	void StereoFrontend::printReprojectionReport(const ReprojectionReport& r)
	{
		const std::size_t total = r.stereoObservations + r.monoObservations;
		auto line = [](const char* name, const ReprojectionGroupStats& s) {
			std::cout << "  " << std::left << std::setw(16) << name << std::right
					  << std::setw(7) << s.count << " obs | median " << std::fixed << std::setprecision(2)
					  << std::setw(6) << s.medianPixels << " px | p90 " << std::setw(6) << s.p90Pixels
					  << " px | >5px " << std::setprecision(1) << std::setw(5) << 100.0 * s.fractionAbove5Pixels
					  << " % | chi2 outliers " << std::setw(5) << 100.0 * s.fractionChi2Outliers << " %\n";
		};

		std::cout << "---- Reprojection report (last " << r.keyframesChecked << " keyframes) ----\n";
		std::cout << "  observations   " << total << " (stereo " << r.stereoObservations
				  << ", mono " << r.monoObservations << "), behind camera " << r.behindCamera << '\n';
		line("anchor", r.anchor);
		line("re-observations", r.reobserved);
		std::cout << "----------------------------------------------------------\n";
	}

	// ------------------------------------------------ dense ICP (parallel)

	// The new keyframe's dense cloud vs the previous keyframe that has one.
	// Copies of both clouds and the tracker's relative pose go to a worker
	// thread, so tracking does not wait; the result is picked up at a later
	// local BA (the new keyframe stays in the BA window for ~10 keyframes).
	void StereoFrontend::launchDenseIcp(std::size_t newKeyframeId)
	{
		const auto source = denseClouds_.find(newKeyframeId);
		if (source == denseClouds_.end() || source->second.pointsCamera.empty()) return;
		for (std::size_t back = 1; back <= 2 * std::max<std::size_t>(1, denseOptions_.everyNthKeyframe) &&
								   back <= newKeyframeId;
			 ++back) {
			const std::size_t a = newKeyframeId - back;
			const auto target = denseClouds_.find(a);
			if (target == denseClouds_.end() || target->second.pointsCamera.empty()) continue;

			const Eigen::Isometry3d initial = keyframes_[a].worldFromCamera.inverse() * keyframes_[newKeyframeId].worldFromCamera;
			PendingIcp job;
			job.keyframeA = a;
			job.keyframeB = newKeyframeId;
			job.weakTracking = icpOnlyBelowInliers_ == 0 || stats_.localInliers < icpOnlyBelowInliers_;
			job.result = std::async(std::launch::async, alignPointToPlane, target->second.pointsCamera,
									source->second.pointsCamera, initial, icpOptions_);  // arguments are copied
			pendingIcp_.push_back(std::move(job));
			++icpLaunched_;
			return;
		}
	}

	void StereoFrontend::collectFinishedIcp(bool wait)
	{
		for (auto it = pendingIcp_.begin(); it != pendingIcp_.end();) {
			if (!wait && it->result.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
				++it;
				continue;
			}
			const IcpResult r = it->result.get();
			icpMillisecondsSum_ += r.milliseconds;
			if (r.valid && !it->weakTracking) {
				++icpRejections_["not needed (tracking was strong)"];
			} else if (r.valid) {
				RelativePoseConstraint c;
				c.keyframeA = it->keyframeA;
				c.keyframeB = it->keyframeB;
				c.aFromB = r.targetFromSource;
				c.information = r.information;
				icpConstraints_.push_back(c);
				icpRmseSum_ += r.rmse;
			} else {
				++icpRejections_[r.reason];
			}
			it = pendingIcp_.erase(it);
		}
	}

	void StereoFrontend::printIcpSummary()
	{
		collectFinishedIcp(true);
		const std::size_t finished = icpConstraints_.size() +
			std::accumulate(icpRejections_.begin(), icpRejections_.end(), std::size_t{0},
							[](std::size_t sum, const auto& kv) { return sum + kv.second; });
		std::cout << std::fixed << std::setprecision(1) << "Dense ICP: " << icpLaunched_ << " launched, "
				  << icpConstraints_.size() << " accepted as BA edges";
		if (!icpConstraints_.empty()) std::cout << " (mean rmse " << 100.0 * icpRmseSum_ / icpConstraints_.size() << " cm)";
		if (finished > 0) std::cout << ", " << icpMillisecondsSum_ / finished << " ms per alignment on a worker thread";
		std::cout << '\n';
		for (const auto& [reason, count] : icpRejections_) std::cout << "  rejected: " << reason << " x" << count << '\n';
	}

	// ------------------------------------------------ local bundle adjustment

	void StereoFrontend::runLocalBundleAdjustmentOnNewKeyframe(const StereoFrame& frame)
	{
		if (!enableLocalBundleAdjustment_) return;

		const bool saveDebug = bundleAdjustmentDebugEveryKeyframes_ > 0 &&
							   keyframes_.size() % bundleAdjustmentDebugEveryKeyframes_ == 0;
		std::vector<ObservationProjection> before;
		if (saveDebug) before = projectNewestKeyframeReobservations();

		collectFinishedIcp();
		lastBundleAdjustment_ = runLocalBundleAdjustment(keyframes_, mapPoints_, calibration_,
														 localBundleAdjustmentOptions_, &icpConstraints_);
		const BundleAdjustmentSummary& ba = lastBundleAdjustment_;
		if (!ba.ran) return;

		// The current frame IS the newest keyframe, so take over its corrected pose.
		worldFromCamera_ = keyframes_.back().worldFromCamera;

		stats_.bundleAdjustmentRan = true;
		stats_.bundleAdjustmentMilliseconds = ba.milliseconds;
		stats_.bundleAdjustmentOutliers = ba.outliersRemoved;
		stats_.bundleAdjustmentRmsBefore = ba.rmsNormalizedBefore;
		stats_.bundleAdjustmentRmsAfter = ba.rmsNormalizedAfter;
		stats_.icpEdgesInBundleAdjustment = ba.relativeConstraints - ba.relativeOutliers;

		std::cout << std::fixed << std::setprecision(2)
				  << "LocalBA kf " << keyframes_.back().id << ": free " << ba.freeKeyframes
				  << " fixed " << ba.fixedKeyframes << " | points " << ba.points << " obs " << ba.observations
				  << " | rms " << ba.rmsNormalizedBefore << " -> " << ba.rmsNormalizedAfter << " sigma"
				  << " | outliers " << ba.outliersRemoved << " | icp " << (ba.relativeConstraints - ba.relativeOutliers)
				  << " | " << ba.iterations << " it, "
				  << std::setprecision(1) << ba.milliseconds << " ms\n";

		if (saveDebug) {
			saveBundleAdjustmentDebugImage(frame.left.image, before, projectNewestKeyframeReobservations());
		}
	}

	// Observations of the newest keyframe whose point was created EARLIER (not
	// the anchor, which is ~0 px by construction): the ones BA can improve.
	std::vector<StereoFrontend::ObservationProjection> StereoFrontend::projectNewestKeyframeReobservations() const
	{
		std::vector<ObservationProjection> projections;
		if (keyframes_.empty()) return projections;

		const Keyframe& keyframe = keyframes_.back();
		const Eigen::Isometry3d cameraFromWorld = keyframe.worldFromCamera.inverse();

		for (std::size_t k = 0; k < keyframe.mapPointIds.size(); ++k) {
			const int pointIndex = keyframe.mapPointIds[k];
			if (pointIndex < 0) continue;
			const SparseMapPoint& point = mapPoints_.at(static_cast<std::size_t>(pointIndex));
			if (!point.valid || point.observations.empty() || point.observations.front().keyframeId == keyframe.id) {
				continue;  // anchor observation: created by this keyframe
			}

			const Eigen::Vector3d pointCamera = cameraFromWorld * point.positionWorld;
			if (pointCamera.z() <= 0.0) continue;

			ObservationProjection projection;
			projection.pointIndex = static_cast<std::size_t>(pointIndex);
			projection.keypointIndex = static_cast<int>(k);
			projection.measured = keyframe.keypoints[k].pt;
			projection.predicted = cv::Point2f(
				static_cast<float>(calibration_.left.fx * pointCamera.x() / pointCamera.z() + calibration_.left.cx),
				static_cast<float>(calibration_.left.fy * pointCamera.y() / pointCamera.z() + calibration_.left.cy));
			projections.push_back(projection);
		}
		return projections;
	}

	// Measured keypoints (green) with their reprojection error drawn 10x enlarged:
	// red = before BA, blue = after BA, magenta ring = observation removed as outlier.
	// If BA works, the blue lines are clearly shorter than the red ones.
	void StereoFrontend::saveBundleAdjustmentDebugImage(const cv::Mat& image,
														const std::vector<ObservationProjection>& before,
														const std::vector<ObservationProjection>& after) const
	{
		if (image.empty() || before.empty()) return;

		constexpr float drawScale = 2.0F;   // upscale the image for readability
		constexpr float errorScale = 10.0F; // enlarge error vectors
		cv::Mat canvas;
		if (image.channels() == 1) cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
		else canvas = image.clone();
		cv::resize(canvas, canvas, cv::Size(), drawScale, drawScale, cv::INTER_LINEAR);

		std::unordered_map<std::size_t, cv::Point2f> afterByPoint;
		for (const ObservationProjection& p : after) afterByPoint[p.pointIndex] = p.predicted;

		std::vector<double> errorsBefore, errorsAfter;
		std::size_t removed = 0;
		const Keyframe& keyframe = keyframes_.back();

		for (const ObservationProjection& b : before) {
			const cv::Point2f measured = b.measured * drawScale;
			const cv::Point2f errorBefore = b.predicted - b.measured;
			errorsBefore.push_back(std::hypot(errorBefore.x, errorBefore.y));
			cv::line(canvas, measured, measured + errorBefore * (errorScale * drawScale),
					 cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

			const bool stillObserved =
				keyframe.mapPointIds.at(static_cast<std::size_t>(b.keypointIndex)) == static_cast<int>(b.pointIndex);
			const auto it = afterByPoint.find(b.pointIndex);
			if (!stillObserved || it == afterByPoint.end()) {
				cv::circle(canvas, measured, 6, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
				++removed;
			} else {
				const cv::Point2f errorAfter = it->second - b.measured;
				errorsAfter.push_back(std::hypot(errorAfter.x, errorAfter.y));
				cv::line(canvas, measured, measured + errorAfter * (errorScale * drawScale),
						 cv::Scalar(255, 128, 0), 1, cv::LINE_AA);
			}
			cv::circle(canvas, measured, 2, cv::Scalar(0, 200, 0), cv::FILLED, cv::LINE_AA);
		}

		auto median = [](std::vector<double> values) {
			if (values.empty()) return 0.0;
			std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
			return values[values.size() / 2];
		};

		std::ostringstream title;
		title << std::fixed << std::setprecision(2) << "keyframe " << keyframe.id << " | " << before.size()
			  << " re-observations | median error before " << median(errorsBefore) << " px, after "
			  << median(errorsAfter) << " px | removed " << removed
			  << " | red = before, blue = after (errors x10), magenta = outlier";
		cv::rectangle(canvas, {0, 0}, {canvas.cols, 34}, cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(canvas, title.str(), {10, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
					cv::LINE_AA);

		std::error_code error;
		std::filesystem::create_directories(bundleAdjustmentDebugDirectory_, error);
		std::ostringstream path;
		path << bundleAdjustmentDebugDirectory_ << "/kf_" << std::setw(6) << std::setfill('0') << keyframe.id << ".png";
		cv::imwrite(path.str(), canvas);
	}

	// ------------------------------------------------------------- frontend

    StereoFrontend::StereoFrontend(const StereoCalibration& calibration)
        : calibration_(calibration)
	{
		// Detect candidateMultiplier_ x more corners than we keep, at a low FAST
		// threshold so weak-texture regions still produce some. distributeOnGrid()
		// then keeps an even spread of targetFeatures_, and only those get descriptors.
		const int candidateCount = targetFeatures_ * candidateMultiplier_;
		orbLeft_ = cv::ORB::create(candidateCount, 1.2F, 8, 31, 0, 2,
								   cv::ORB::HARRIS_SCORE, 31, fastThreshold_);
		orbRight_ = cv::ORB::create(candidateCount, 1.2F, 8, 31, 0, 2,
									cv::ORB::HARRIS_SCORE, 31, fastThreshold_);

		cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
		cameraMatrix_.at<double>(0, 0) = calibration_.left.fx;
		cameraMatrix_.at<double>(1, 1) = calibration_.left.fy;
		cameraMatrix_.at<double>(0, 2) = calibration_.left.cx;
		cameraMatrix_.at<double>(1, 2) = calibration_.left.cy;

		// KITTI stereo images are rectified.
		distortionCoefficients_ = cv::Mat::zeros(4, 1, CV_64F);

		// ORB-SLAM's "close point" depth: 35 baselines (~19 m on KITTI).
		closeDepthMeters_ = 35.0 * calibration_.baseline_m;

		std::cout << "Frontend is created\n";
	}

	bool StereoFrontend::process(const StereoFrame& frame)
	{
		const bool ok = processFrame(frame);

		// Remember the pose relative to the newest keyframe, so loop corrections
		// of that keyframe later also correct this frame in finalTrajectory().
		FrameRecord record;
		if (!keyframes_.empty()) {
			record.referenceKeyframe = static_cast<long>(keyframes_.back().id);
			record.keyframeFromCamera = keyframes_.back().worldFromCamera.inverse() * worldFromCamera_;
		} else {
			record.keyframeFromCamera = worldFromCamera_;
		}
		frameRecords_.push_back(record);
		return ok;
	}

	std::vector<Eigen::Isometry3d> StereoFrontend::finalTrajectory() const
	{
		std::vector<Eigen::Isometry3d> trajectory;
		trajectory.reserve(frameRecords_.size());
		for (const FrameRecord& record : frameRecords_) {
			if (record.referenceKeyframe < 0) {
				trajectory.push_back(record.keyframeFromCamera);
			} else {
				trajectory.push_back(keyframes_.at(static_cast<std::size_t>(record.referenceKeyframe)).worldFromCamera *
									 record.keyframeFromCamera);
			}
		}
		return trajectory;
	}

	bool StereoFrontend::processFrame(const StereoFrame& frame) // Process engine
	{
		stats_ = FrameStats{};
		stats_.frameIndex = frameIndex_++;
		stats_.timestamp_s = frame.timestamp_s;
		currentLeftImage_ = frame.left.image;

		extractCurrentFeatures(frame);
		stats_.features = currentLeft_.keypoints.size();

		// Constant-velocity motion model: assume the last frame-to-frame motion repeats.
		predictedWorldFromCamera_ = worldFromCamera_ * velocity_;

		if (currentLeft_.descriptors.empty() || currentRight_.descriptors.empty()) {
			std::cout << "No descriptors in current stereo frame\n";
			stats_.path = "no_desc";
			worldFromCamera_ = predictedWorldFromCamera_;
			return false;
		}

		const std::size_t stereoMatchCount = findCurrentStereoMatches();
		stats_.stereoMatches = stereoMatchCount;

		if (!mapInitialized_) { // Initialize the map when there are enough points
			worldFromCamera_ = Eigen::Isometry3d::Identity();
			const std::size_t initialPointCount = addNewMapPointsFromCurrentStereo();
			mapInitialized_ = initialPointCount > 20;
			frontendInitialized_ = mapInitialized_;

			if (mapInitialized_) {
				framesSinceLastKeyframe_ = 0;
				saveCurrentAsKeyframe(frame, initialPointCount);
				std::cout << "Initial sparse map points: " << initialPointCount << '\n';
			}

			stats_.path = mapInitialized_ ? "init" : "init_fail";
			stats_.keyframeInserted = mapInitialized_;
			stats_.keyframeReason = mapInitialized_ ? "init" : "";
			assignNewToPrevious(frame);
			return mapInitialized_;
		}

		// Both trackers read worldFromCamera_ as the PREVIOUS pose, so keep it
		// untouched until one of them succeeds.
		const Eigen::Isometry3d previousWorldFromCamera = worldFromCamera_;

		// Normal tracker: only the last N keyframes' landmarks. It needs no
		// current stereo, so it runs even when stereo matching is weak.
		const bool usedLocalMapTracking = trackFromLocalKeyframes();

		bool usedPreviousStereoTracking = false;
		if (!usedLocalMapTracking) {
			std::cout << "Local-keyframe PnP failed; trying previous stereo PnP...\n";
			usedPreviousStereoTracking = trackFromPreviousStereo();
		}

		if (usedLocalMapTracking || usedPreviousStereoTracking) {
			stats_.path = usedLocalMapTracking ? "local" : "prev_stereo";
			velocity_ = previousWorldFromCamera.inverse() * worldFromCamera_;

			// Count associations before a keyframe adds new points, so this reflects
			// how many existing map points the frame actually re-observed.
			const std::size_t trackedMapPoints = static_cast<std::size_t>(std::count_if(
				currentLeft_.mapPointIds.begin(), currentLeft_.mapPointIds.end(),
				[](const int id) { return id >= 0; }));
			stats_.trackedMapPoints = trackedMapPoints;

			++framesSinceLastKeyframe_;
			// The first frame after a keyframe sets the reference: how much of the
			// map a fresh view explains. Later frames are compared against it.
			if (framesSinceLastKeyframe_ == 1) referenceTracked_ = trackedMapPoints;

			const char* reason = keyframeReason(trackedMapPoints, usedPreviousStereoTracking);
			keyframeDetected_ = reason != nullptr;
			if (keyframeDetected_) {
				const std::size_t addedPointCount = addNewMapPointsFromCurrentStereo();
				saveCurrentAsKeyframe(frame, addedPointCount);
				stats_.keyframeInserted = true;
				stats_.keyframeReason = reason;

				// Refine the window around the new keyframe. It may move the current
				// pose, so the motion model is recomputed from the corrected pose.
				runLocalBundleAdjustmentOnNewKeyframe(frame);
				velocity_ = previousWorldFromCamera.inverse() * worldFromCamera_;

				// Loop closure last: it may move the whole map, including the current
				// pose. velocity_ is relative, so it stays valid after the correction.
				detectAndCloseLoop();
			}

			// This must happen after association transfer and possible keyframe save.
			assignNewToPrevious(frame);
			return true;
		}

		// Both metric trackers failed. Never freeze: move on with the motion model
		// and, if this frame has enough stereo, restart the local map from it.
		worldFromCamera_ = predictedWorldFromCamera_;
		keyframeDetected_ = false;

		if (stereoMatchCount >= minStereoForReinit_) {
			// Old local keyframes no longer match the view; drop them from the window
			// (they stay in keyframes_ / mapPoints_ for later loop closure).
			localKeyframeIndices_.clear();
			const std::size_t addedPointCount = addNewMapPointsFromCurrentStereo();
			saveCurrentAsKeyframe(frame, addedPointCount);
			keyframeDetected_ = true;
			stats_.path = "reinit";
			stats_.keyframeInserted = true;
			stats_.keyframeReason = "reinit";
			std::cout << "Tracking lost: re-initialized local map at the predicted pose with "
					  << addedPointCount << " points.\n";
		} else {
			stats_.path = "lost";
			std::cout << "Tracking lost and too few stereo points to re-initialize; using motion model.\n";
		}

		// Always advance the previous frame. Comparing every new frame against a
		// stale one is what made the old version stay stuck forever.
		assignNewToPrevious(frame);
		return keyframeDetected_;
	}

	// ----------------------------------------------------- feature extraction

	void StereoFrontend::extractCurrentFeatures(const StereoFrame& frame)
	{
		imageWidth_ = frame.left.image.cols;
		imageHeight_ = frame.left.image.rows;

		detectDistributed(orbLeft_, frame.left.image, currentLeft_);
		detectDistributed(orbRight_, frame.right.image, currentRight_);

		currentLeft_.mapPointIds.assign(currentLeft_.keypoints.size(), -1);
		currentRight_.mapPointIds.assign(currentRight_.keypoints.size(), -1);
	}

	void StereoFrontend::detectDistributed(const cv::Ptr<cv::ORB>& orb, const cv::Mat& image,
										   ImageFeatures& features) const
	{
		std::vector<cv::KeyPoint> candidates;
		orb->detect(image, candidates);
		features.keypoints = distributeOnGrid(candidates, image.cols, image.rows);

		// compute() may drop keypoints too close to the border; it updates the vector.
		orb->compute(image, features.keypoints, features.descriptors);
	}

	// Bucketing: split the image into a grid, keep the strongest few corners per
	// cell, then spend the budget empty cells (sky, road) left on the strongest
	// remaining corners. Prevents all features clustering on one textured patch.
	std::vector<cv::KeyPoint> StereoFrontend::distributeOnGrid(
		std::vector<cv::KeyPoint>& candidates, const int width, const int height) const
	{
		const std::size_t target = static_cast<std::size_t>(targetFeatures_);
		if (candidates.size() <= target || width <= 0 || height <= 0) return candidates;

		const int cellCount = detectionGridRows_ * detectionGridCols_;
		const std::size_t perCell = std::max<std::size_t>(1, target / static_cast<std::size_t>(cellCount));

		std::sort(candidates.begin(), candidates.end(),
				  [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });

		std::vector<std::size_t> usedPerCell(static_cast<std::size_t>(cellCount), 0);
		std::vector<cv::KeyPoint> kept;
		std::vector<cv::KeyPoint> leftover;
		kept.reserve(target);

		for (const cv::KeyPoint& keypoint : candidates) {
			const int col = std::clamp(static_cast<int>(keypoint.pt.x * detectionGridCols_ / width),
									   0, detectionGridCols_ - 1);
			const int row = std::clamp(static_cast<int>(keypoint.pt.y * detectionGridRows_ / height),
									   0, detectionGridRows_ - 1);
			std::size_t& used = usedPerCell[static_cast<std::size_t>(row * detectionGridCols_ + col)];
			if (used < perCell) {
				kept.push_back(keypoint);
				++used;
			} else {
				leftover.push_back(keypoint);
			}
		}

		for (const cv::KeyPoint& keypoint : leftover) {
			if (kept.size() >= target) break;
			kept.push_back(keypoint);
		}
		return kept;
	}

	// ------------------------------------------------------- stereo matching

	// Rectified stereo: a correct match lies on the same row, at a disparity set
	// by depth (d = fx * b / Z). Search only there, so look-alikes on other rows
	// can never compete. ORB-SLAM2's ComputeStereoMatches without the SAD refinement.
	std::size_t StereoFrontend::findCurrentStereoMatches()
	{
		currentStereoMatches_.clear();
		if (currentLeft_.descriptors.empty() || currentRight_.descriptors.empty() || imageHeight_ <= 0) {
			return 0;
		}

		const int rowCount = imageHeight_;
		const std::size_t rightCount = currentRight_.keypoints.size();

		// Row index: each right keypoint is listed in every row its position
		// uncertainty covers. Coarser pyramid levels are less precise.
		std::vector<std::vector<int>> rightIndicesByRow(static_cast<std::size_t>(rowCount));
		for (std::size_t r = 0; r < rightCount; ++r) {
			const cv::KeyPoint& keypoint = currentRight_.keypoints[r];
			const float tolerance = maximumEpipolarErrorPixels_ *
									std::pow(1.2F, static_cast<float>(keypoint.octave));
			const int minRow = std::max(0, static_cast<int>(std::floor(keypoint.pt.y - tolerance)));
			const int maxRow = std::min(rowCount - 1, static_cast<int>(std::ceil(keypoint.pt.y + tolerance)));
			for (int row = minRow; row <= maxRow; ++row) {
				rightIndicesByRow[static_cast<std::size_t>(row)].push_back(static_cast<int>(r));
			}
		}

		// Depth limits -> disparity window.
		const double focalTimesBaseline = calibration_.left.fx * calibration_.baseline_m;
		const float minDisparity = std::max(minimumDisparityPixels_,
											static_cast<float>(focalTimesBaseline / maximumDepthMeters_));
		const float maxDisparity = static_cast<float>(focalTimesBaseline / minimumDepthMeters_);

		// Best left candidate per right keypoint (enforces one-to-one matching
		// by keeping the closest descriptor, not whichever came first).
		std::vector<int> bestLeftForRight(rightCount, -1);
		std::vector<int> bestDistanceForRight(rightCount, INT_MAX);

		for (std::size_t l = 0; l < currentLeft_.keypoints.size(); ++l) {
			const cv::KeyPoint& left = currentLeft_.keypoints[l];
			const int row = std::clamp(static_cast<int>(std::lround(left.pt.y)), 0, rowCount - 1);
			const cv::Mat leftDescriptor = currentLeft_.descriptors.row(static_cast<int>(l));

			int bestDistance = INT_MAX;
			int bestRight = -1;
			for (const int r : rightIndicesByRow[static_cast<std::size_t>(row)]) {
				const cv::KeyPoint& right = currentRight_.keypoints[static_cast<std::size_t>(r)];
				// The same point should be detected at a similar scale in both images.
				if (std::abs(right.octave - left.octave) > 1) continue;

				const float disparity = left.pt.x - right.pt.x;
				if (disparity < minDisparity || disparity > maxDisparity) continue;

				const int distance = static_cast<int>(cv::norm(
					leftDescriptor, currentRight_.descriptors.row(r), cv::NORM_HAMMING));
				if (distance < bestDistance) {
					bestDistance = distance;
					bestRight = r;
				}
			}

			if (bestRight < 0 || bestDistance > maximumStereoDescriptorDistance_) continue;
			if (bestDistance < bestDistanceForRight[static_cast<std::size_t>(bestRight)]) {
				bestDistanceForRight[static_cast<std::size_t>(bestRight)] = bestDistance;
				bestLeftForRight[static_cast<std::size_t>(bestRight)] = static_cast<int>(l);
			}
		}

		// Collect unique pairs, then reject descriptor-distance outliers relative
		// to this frame's median (ORB-SLAM2 uses 1.5 * 1.4 * median).
		std::vector<std::pair<int, int>> pairs;  // (left, right)
		std::vector<int> distances;
		for (std::size_t r = 0; r < rightCount; ++r) {
			if (bestLeftForRight[r] < 0) continue;
			pairs.emplace_back(bestLeftForRight[r], static_cast<int>(r));
			distances.push_back(bestDistanceForRight[r]);
		}
		if (pairs.empty()) return 0;

		std::vector<int> sortedDistances = distances;
		std::nth_element(sortedDistances.begin(), sortedDistances.begin() + sortedDistances.size() / 2,
						 sortedDistances.end());
		const double distanceThreshold = 1.5 * 1.4 * sortedDistances[sortedDistances.size() / 2];

		for (std::size_t i = 0; i < pairs.size(); ++i) {
			if (distances[i] > distanceThreshold) continue;

			const int leftIndex = pairs[i].first;
			const int rightIndex = pairs[i].second;
			const cv::KeyPoint& left = currentLeft_.keypoints.at(static_cast<std::size_t>(leftIndex));
			const cv::KeyPoint& right = currentRight_.keypoints.at(static_cast<std::size_t>(rightIndex));

			const float disparity = left.pt.x - right.pt.x;
			const double depth = focalTimesBaseline / static_cast<double>(disparity);
			const double x = (left.pt.x - calibration_.left.cx) * depth / calibration_.left.fx;
			const double y = (left.pt.y - calibration_.left.cy) * depth / calibration_.left.fy;

			currentStereoMatches_.push_back({leftIndex, rightIndex, static_cast<float>(distances[i]),
											 disparity, Eigen::Vector3d(x, y, depth)});
		}

		std::cout << "Valid stereo matches: " << currentStereoMatches_.size() << '\n';
		return currentStereoMatches_.size();
	}

	std::size_t StereoFrontend::addNewMapPointsFromCurrentStereo()
	{
		std::size_t addedPointCount = 0;
		for (const StereoMatch& match : currentStereoMatches_) {
			const int leftIndex = match.leftKeypointIndex;
			if (leftIndex < 0 || static_cast<std::size_t>(leftIndex) >= currentLeft_.mapPointIds.size()) {
				continue;
			}
			// Tracking already associated this feature with an old landmark.
			if (currentLeft_.mapPointIds.at(leftIndex) >= 0) continue;

			SparseMapPoint mapPoint;
			mapPoint.id = nextMapPointId_++;
			mapPoint.positionWorld = worldFromCamera_ * match.pointLeftCamera;
			mapPoint.descriptor = currentLeft_.descriptors.row(leftIndex).clone();
			mapPoint.firstKeyframeId = nextKeyframeId_;  // the keyframe about to be saved
			if (!currentLeftImage_.empty() && currentLeftImage_.type() == CV_8UC1) {
				const cv::Point2f& pt = currentLeft_.keypoints.at(static_cast<std::size_t>(leftIndex)).pt;
				mapPoint.intensity = currentLeftImage_.at<std::uint8_t>(
					std::clamp(static_cast<int>(pt.y), 0, currentLeftImage_.rows - 1),
					std::clamp(static_cast<int>(pt.x), 0, currentLeftImage_.cols - 1));
			}

			const int mapPointIndex = static_cast<int>(mapPoints_.size());
			mapPoints_.push_back(std::move(mapPoint));
			currentLeft_.mapPointIds.at(leftIndex) = mapPointIndex;
			recentMapPoints_.push_back(static_cast<std::size_t>(mapPointIndex));
			++addedPointCount;
		}
		return addedPointCount;
	}

	// ------------------------------------------------------------- keyframes

	// Insert when tracking quality drops, with time only as an upper bound.
	// Simplified from ORB-SLAM2's Tracking::NeedNewKeyFrame (no mapping thread).
	const char* StereoFrontend::keyframeReason(const std::size_t trackedMapPoints,
											   const bool usedFallback) const
	{
		if (framesSinceLastKeyframe_ < minKeyframeIntervalFrames_) return nullptr;

		// Local map just failed: refresh it from this frame right away.
		if (usedFallback) return "fallback";

		if (framesSinceLastKeyframe_ >= maxKeyframeIntervalFrames_) return "max_interval";

		if (trackedMapPoints < minTrackedForKeyframe_) return "low_tracked";

		// The view has drifted away from what the map explains.
		if (framesSinceLastKeyframe_ >= 2 && referenceTracked_ > 0 &&
			static_cast<double>(trackedMapPoints) <
				keyframeTrackedRatio_ * static_cast<double>(referenceTracked_)) {
			return "ratio";
		}

		// Stereo-specific: few accurate close points are tracked, but many new
		// close points are visible. Typical at turn onset.
		if (!useCloseKeyframeRule_) return nullptr;
		std::size_t closeTracked = 0;
		std::size_t closeUntracked = 0;
		for (const StereoMatch& match : currentStereoMatches_) {
			if (match.pointLeftCamera.z() >= closeDepthMeters_) continue;
			if (currentLeft_.mapPointIds.at(match.leftKeypointIndex) >= 0) ++closeTracked;
			else ++closeUntracked;
		}
		if (closeTracked < closeTrackedMax_ && closeUntracked > closeUntrackedMin_) return "close";

		return nullptr;
	}

	void StereoFrontend::saveCurrentAsKeyframe(const StereoFrame& frame,
												const std::size_t addedPointCount)
	{
		Keyframe keyframe;
		keyframe.id = nextKeyframeId_++;
		keyframe.timestamp_s = frame.timestamp_s;
		keyframe.frameIndex = stats_.frameIndex;
		keyframe.worldFromCamera = worldFromCamera_;
		keyframe.keypoints = currentLeft_.keypoints;
		keyframe.descriptors = currentLeft_.descriptors.clone();
		keyframe.mapPointIds = currentLeft_.mapPointIds;

		// Right-image x for every left keypoint that has a stereo match in THIS
		// frame. Gives the third residual row (u_R) in bundle adjustment.
		std::vector<double> rightXByLeftIndex(currentLeft_.keypoints.size(), -1.0);
		for (const StereoMatch& match : currentStereoMatches_) {
			rightXByLeftIndex.at(static_cast<std::size_t>(match.leftKeypointIndex)) =
				currentRight_.keypoints.at(static_cast<std::size_t>(match.rightKeypointIndex)).pt.x;
		}

		for (std::size_t keypointIndex = 0;
				keypointIndex < keyframe.mapPointIds.size(); ++keypointIndex) {
			const int mapPointIndex = keyframe.mapPointIds.at(keypointIndex);
			if (mapPointIndex < 0 || static_cast<std::size_t>(mapPointIndex) >= mapPoints_.size()) {
				continue;
			}
			const cv::KeyPoint& keypoint = keyframe.keypoints.at(keypointIndex);

			Observation observation;
			observation.keyframeId = keyframe.id;
			observation.keypointIndex = static_cast<int>(keypointIndex);
			observation.pixel = Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y);
			observation.uRight = rightXByLeftIndex.at(keypointIndex);
			observation.octave = keypoint.octave;
			mapPoints_.at(mapPointIndex).observations.push_back(observation);
		}

		// Keyframe ids are their index in keyframes_ (keyframes are never erased).
		// reprojectionReport() and bundle adjustment rely on this; keyframe culling
		// would need an id -> index map instead.
		assert(keyframe.id == keyframes_.size());
		keyframes_.push_back(std::move(keyframe));
		updateCovisibility(keyframes_.back().id);
		if (useMapPointCulling_) stats_.culledPoints = cullRecentMapPoints();

		// Dense stereo cloud, stored in this keyframe's camera frame so BA and loop
		// closure move it with the keyframe.
		if (denseOptions_.enabled && denseOptions_.everyNthKeyframe > 0 &&
			keyframes_.back().id % denseOptions_.everyNthKeyframe == 0) {
			const auto denseStart = std::chrono::steady_clock::now();
			denseClouds_[keyframes_.back().id] =
				computeDenseCloud(frame.left.image, frame.right.image, calibration_, denseOptions_, frame.color);
			stats_.denseMilliseconds = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - denseStart).count();
			if (useDenseIcp_) launchDenseIcp(keyframes_.back().id);
		}
		localKeyframeIndices_.push_back(keyframes_.size() - 1);
		if (localKeyframeIndices_.size() > localTrackingKeyframeCount_) {
			localKeyframeIndices_.pop_front();
		}

		framesSinceLastKeyframe_ = 0;
		referenceTracked_ = 0;  // set again on the next tracked frame

		std::cout << "Keyframe " << keyframes_.back().id
					<< " saved. Added map points: " << addedPointCount
					<< " | Total keyframes: " << keyframes_.size()
					<< " | Local tracking keyframes: " << localKeyframeIndices_.size()
					<< " | Total map points: " << mapPoints_.size() << '\n';

		if (reprojectionReportEveryKeyframes_ > 0 &&
			keyframes_.size() % reprojectionReportEveryKeyframes_ == 0) {
			printReprojectionReport(reprojectionReport());
		}
	}

	void StereoFrontend::assignNewToPrevious(const StereoFrame& frame)
	{
		previousFrame_ = frame;
		previousLeft_ = currentLeft_;
		previousRight_ = currentRight_;

		// cv::Mat assignment is shallow: without clone() both frames share one
		// buffer, and the next detectAndCompute()/compute() overwrites the
		// "previous" descriptors in place whenever the row count is unchanged.
		previousLeft_.descriptors = currentLeft_.descriptors.clone();
		previousRight_.descriptors = currentRight_.descriptors.clone();

		previousStereoMatches_ = currentStereoMatches_;
	}

	// -------------------------------------------------------------- tracking

	// ------------------------------------------------------ covisibility graph

	// Two keyframes are connected when they observe the same map points; the
	// edge weight is how many. Called once per new keyframe, after its
	// observations were added. (Weights are not reduced when BA removes outlier
	// observations later; good enough for choosing neighbours.)
	void StereoFrontend::updateCovisibility(const std::size_t keyframeId)
	{
		if (covisibilityWeights_.size() <= keyframeId) covisibilityWeights_.resize(keyframeId + 1);

		std::unordered_map<std::size_t, int> shared;
		for (const int pointIndex : keyframes_.at(keyframeId).mapPointIds) {
			if (pointIndex < 0) continue;
			const SparseMapPoint& point = mapPoints_.at(static_cast<std::size_t>(pointIndex));
			if (!point.valid) continue;
			for (const Observation& observation : point.observations) {
				if (observation.keyframeId != keyframeId) ++shared[observation.keyframeId];
			}
		}

		for (const auto& [otherId, weight] : shared) {
			covisibilityWeights_[keyframeId][otherId] = weight;
			covisibilityWeights_.at(otherId)[keyframeId] = weight;  // undirected edge
		}
	}

	std::vector<std::size_t> StereoFrontend::bestCovisibleKeyframes(const std::size_t keyframeId,
																	 const std::size_t count,
																	 const int minWeight) const
	{
		std::vector<std::pair<int, std::size_t>> neighbours;  // (weight, id)
		if (keyframeId >= covisibilityWeights_.size()) return {};
		for (const auto& [otherId, weight] : covisibilityWeights_[keyframeId]) {
			if (weight >= minWeight) neighbours.emplace_back(weight, otherId);
		}
		std::sort(neighbours.begin(), neighbours.end(), [](const auto& a, const auto& b) {
			return a.first != b.first ? a.first > b.first : a.second > b.second;  // strongest, then newest
		});

		std::vector<std::size_t> result;
		for (std::size_t i = 0; i < neighbours.size() && i < count; ++i) result.push_back(neighbours[i].second);
		return result;
	}

	// Newest keyframe + its best covisible keyframes (the ones that actually see
	// the same scene). Falls back to "last N by time" when the graph is off.
	std::vector<std::size_t> StereoFrontend::selectLocalKeyframes() const
	{
		if (!useCovisibilityLocalMap_ || localKeyframeIndices_.empty()) {
			return {localKeyframeIndices_.begin(), localKeyframeIndices_.end()};
		}

		const std::size_t reference = localKeyframeIndices_.back();  // newest keyframe of the current map segment
		std::vector<std::size_t> selected{reference};
		for (const std::size_t id : bestCovisibleKeyframes(reference, localTrackingKeyframeCount_ - 1,
														   covisibilityMinWeight_)) {
			selected.push_back(id);
		}
		return selected;
	}

	// ------------------------------------------------- map point management

	// Second, tight projection search with the SOLVED pose (ORB-SLAM's
	// "track local map"). Only for points and keypoints not matched yet. Every
	// keypoint linked here is one duplicate point that will NOT be created at
	// the next keyframe, and one more observation for bundle adjustment.
	std::size_t StereoFrontend::associateLocalMapAfterPose(const std::vector<int>& localMapPointIndices)
	{
		std::unordered_set<int> alreadyMatched;
		for (const int id : currentLeft_.mapPointIds) {
			if (id >= 0) alreadyMatched.insert(id);
		}
		std::vector<int> candidates;
		candidates.reserve(localMapPointIndices.size());
		for (const int id : localMapPointIndices) {
			if (!alreadyMatched.count(id)) candidates.push_back(id);
		}
		if (candidates.empty()) return 0;

		// Stereo depth per left keypoint, for the depth consistency check.
		std::vector<double> depthByLeftIndex(currentLeft_.keypoints.size(), -1.0);
		for (const StereoMatch& match : currentStereoMatches_) {
			depthByLeftIndex.at(static_cast<std::size_t>(match.leftKeypointIndex)) = match.pointLeftCamera.z();
		}

		const LocalMapMatches matches = matchLocalMapByProjection(candidates, worldFromCamera_, fusionRadiusPixels_);
		const Eigen::Isometry3d cameraFromWorld = worldFromCamera_.inverse();

		std::size_t associated = 0;
		for (std::size_t i = 0; i < matches.size(); ++i) {
			const std::size_t keypoint = static_cast<std::size_t>(matches.keypointIndices[i]);
			const int pointIndex = matches.mapPointIndices[i];
			if (currentLeft_.mapPointIds[keypoint] >= 0) continue;  // keypoint already taken

			// If this keypoint has its own stereo depth, it must agree with the
			// point's predicted depth; otherwise it's a different 3D point.
			const double measuredDepth = depthByLeftIndex[keypoint];
			if (measuredDepth > 0.0) {
				const double predictedDepth =
					(cameraFromWorld * mapPoints_.at(static_cast<std::size_t>(pointIndex)).positionWorld).z();
				if (std::abs(measuredDepth - predictedDepth) > fusionMaxDepthRatioError_ * predictedDepth) continue;
			}

			currentLeft_.mapPointIds[keypoint] = pointIndex;
			++associated;
		}
		return associated;
	}

	void StereoFrontend::updateMapPointStatistics(const std::vector<int>& localMapPointIndices)
	{
		const Eigen::Isometry3d cameraFromWorld = worldFromCamera_.inverse();
		std::unordered_set<int> found;
		for (const int id : currentLeft_.mapPointIds) {
			if (id >= 0) found.insert(id);
		}

		for (const int id : localMapPointIndices) {
			SparseMapPoint& point = mapPoints_.at(static_cast<std::size_t>(id));
			if (!point.valid) continue;
			const Eigen::Vector3d pointCamera = cameraFromWorld * point.positionWorld;
			if (pointCamera.z() <= 0.1) continue;
			const double u = calibration_.left.fx * pointCamera.x() / pointCamera.z() + calibration_.left.cx;
			const double v = calibration_.left.fy * pointCamera.y() / pointCamera.z() + calibration_.left.cy;
			if (u < 0.0 || u >= imageWidth_ || v < 0.0 || v >= imageHeight_) continue;

			++point.visibleCount;
			if (found.count(id)) ++point.foundCount;
		}
	}

	// A point is on probation for a few keyframes after creation. It is removed
	// if it is found in less than 25 % of the frames where it should have been
	// visible, or if after cullAfterKeyframes_ keyframes fewer than
	// cullMinObservations_ keyframes have seen it. Survivors leave probation.
	std::size_t StereoFrontend::cullRecentMapPoints()
	{
		if (keyframes_.empty()) return 0;
		const std::size_t currentKeyframe = keyframes_.back().id;

		std::size_t culled = 0;
		std::vector<std::size_t> stillRecent;
		stillRecent.reserve(recentMapPoints_.size());

		for (const std::size_t index : recentMapPoints_) {
			SparseMapPoint& point = mapPoints_.at(index);
			if (!point.valid) continue;

			const std::size_t age = currentKeyframe >= point.firstKeyframeId ? currentKeyframe - point.firstKeyframeId : 0;
			const bool rarelyFound = point.visibleCount >= cullMinVisibleForRatio_ &&
									 static_cast<double>(point.foundCount) <
										 cullMinFoundRatio_ * static_cast<double>(point.visibleCount);
			const bool tooFewKeyframes = age >= cullAfterKeyframes_ && point.observations.size() < cullMinObservations_;

			if (rarelyFound || tooFewKeyframes) {
				point.valid = false;  // tracking, BA and covisibility all skip invalid points
				++culled;
			} else if (age < cullAfterKeyframes_ + 1) {
				stillRecent.push_back(index);  // keep on probation
			}
		}
		recentMapPoints_ = std::move(stillRecent);
		return culled;
	}

	// ------------------------------------------------------- local map matching

	void StereoFrontend::addLocalMapMatch(LocalMapMatches& matches, const int mapPointIndex,
										  const int keypointIndex) const
	{
		const Eigen::Vector3d& pointWorld = mapPoints_.at(static_cast<std::size_t>(mapPointIndex)).positionWorld;
		matches.objectPoints.emplace_back(static_cast<float>(pointWorld.x()), static_cast<float>(pointWorld.y()),
										  static_cast<float>(pointWorld.z()));
		matches.imagePoints.push_back(currentLeft_.keypoints.at(static_cast<std::size_t>(keypointIndex)).pt);
		matches.mapPointIndices.push_back(mapPointIndex);
		matches.keypointIndices.push_back(keypointIndex);
	}

	// Global matching: every local map point against ALL current keypoints
	// (kNN + ratio test). Needs no pose prediction, so it is the fallback.
	StereoFrontend::LocalMapMatches StereoFrontend::matchLocalMapByDescriptor(
		const std::vector<int>& localMapPointIndices) const
	{
		LocalMapMatches matches;
		cv::Mat localDescriptors;
		std::vector<int> descriptorRowToMapPointIndex;
		for (const int mapPointIndex : localMapPointIndices) {
			const cv::Mat& descriptor = mapPoints_.at(static_cast<std::size_t>(mapPointIndex)).descriptor;
			if (descriptor.empty()) continue;
			localDescriptors.push_back(descriptor);
			descriptorRowToMapPointIndex.push_back(mapPointIndex);
		}
		if (localDescriptors.empty() || currentLeft_.descriptors.empty()) return matches;

		std::vector<std::vector<cv::DMatch>> candidates;
		temporalMatcher_.knnMatch(localDescriptors, currentLeft_.descriptors, candidates, 2);

		std::unordered_set<int> usedCurrentKeypointIndices;
		for (const auto& twoMatches : candidates) {
			if (twoMatches.size() < 2) continue;
			const cv::DMatch& best = twoMatches[0];
			const cv::DMatch& second = twoMatches[1];
			if (best.distance >= 0.75F * second.distance) continue;
			if (!usedCurrentKeypointIndices.insert(best.trainIdx).second) continue;
			addLocalMapMatch(matches, descriptorRowToMapPointIndex.at(static_cast<std::size_t>(best.queryIdx)),
							 best.trainIdx);
		}
		return matches;
	}

	// Projection search: project each local map point with the PREDICTED pose
	// (motion model) and compare descriptors only with keypoints within
	// radiusPixels of where it should appear. Points outside the image are never
	// tried, and look-alikes elsewhere in the image cannot compete.
	StereoFrontend::LocalMapMatches StereoFrontend::matchLocalMapByProjection(
		const std::vector<int>& localMapPointIndices, const Eigen::Isometry3d& predictedWorldFromCamera,
		const float radiusPixels) const
	{
		LocalMapMatches matches;
		if (currentLeft_.keypoints.empty() || imageWidth_ <= 0 || imageHeight_ <= 0) return matches;

		// Keypoint lookup grid, so each query only scans nearby cells.
		const int cell = projectionCellPixels_;
		const int gridCols = (imageWidth_ + cell - 1) / cell;
		const int gridRows = (imageHeight_ + cell - 1) / cell;
		std::vector<std::vector<int>> grid(static_cast<std::size_t>(gridCols * gridRows));
		for (std::size_t k = 0; k < currentLeft_.keypoints.size(); ++k) {
			const cv::Point2f& pt = currentLeft_.keypoints[k].pt;
			const int c = std::clamp(static_cast<int>(pt.x) / cell, 0, gridCols - 1);
			const int r = std::clamp(static_cast<int>(pt.y) / cell, 0, gridRows - 1);
			grid[static_cast<std::size_t>(r * gridCols + c)].push_back(static_cast<int>(k));
		}

		const Eigen::Isometry3d cameraFromWorld = predictedWorldFromCamera.inverse();
		const double fx = calibration_.left.fx, fy = calibration_.left.fy;
		const double cx = calibration_.left.cx, cy = calibration_.left.cy;
		const float radiusSquared = radiusPixels * radiusPixels;

		// One map point per keypoint: keep the closest descriptor.
		std::vector<int> bestPointForKeypoint(currentLeft_.keypoints.size(), -1);
		std::vector<int> bestDistanceForKeypoint(currentLeft_.keypoints.size(), INT_MAX);

		for (const int mapPointIndex : localMapPointIndices) {
			const SparseMapPoint& point = mapPoints_.at(static_cast<std::size_t>(mapPointIndex));
			if (point.descriptor.empty()) continue;

			const Eigen::Vector3d pointCamera = cameraFromWorld * point.positionWorld;
			if (pointCamera.z() <= 0.1) continue;                         // behind the camera
			const float u = static_cast<float>(fx * pointCamera.x() / pointCamera.z() + cx);
			const float v = static_cast<float>(fy * pointCamera.y() / pointCamera.z() + cy);
			if (u < 0.0F || u >= imageWidth_ || v < 0.0F || v >= imageHeight_) continue;  // outside the image

			const int minC = std::max(0, static_cast<int>((u - radiusPixels) / cell));
			const int maxC = std::min(gridCols - 1, static_cast<int>((u + radiusPixels) / cell));
			const int minR = std::max(0, static_cast<int>((v - radiusPixels) / cell));
			const int maxR = std::min(gridRows - 1, static_cast<int>((v + radiusPixels) / cell));

			int bestDistance = INT_MAX, secondDistance = INT_MAX, bestKeypoint = -1;
			for (int r = minR; r <= maxR; ++r) {
				for (int c = minC; c <= maxC; ++c) {
					for (const int k : grid[static_cast<std::size_t>(r * gridCols + c)]) {
						const cv::Point2f& pt = currentLeft_.keypoints[static_cast<std::size_t>(k)].pt;
						const float du = pt.x - u, dv = pt.y - v;
						if (du * du + dv * dv > radiusSquared) continue;

						const int distance = static_cast<int>(
							cv::norm(point.descriptor, currentLeft_.descriptors.row(k), cv::NORM_HAMMING));
						if (distance < bestDistance) {
							secondDistance = bestDistance;
							bestDistance = distance;
							bestKeypoint = k;
						} else if (distance < secondDistance) {
							secondDistance = distance;
						}
					}
				}
			}

			if (bestKeypoint < 0 || bestDistance > projectionMaxDescriptorDistance_) continue;
			if (secondDistance != INT_MAX &&
				static_cast<float>(bestDistance) >= projectionRatio_ * static_cast<float>(secondDistance)) {
				continue;  // ambiguous inside the window
			}
			const std::size_t keypointSlot = static_cast<std::size_t>(bestKeypoint);
			if (bestDistance < bestDistanceForKeypoint[keypointSlot]) {
				bestDistanceForKeypoint[keypointSlot] = bestDistance;
				bestPointForKeypoint[keypointSlot] = mapPointIndex;
			}
		}

		for (std::size_t k = 0; k < bestPointForKeypoint.size(); ++k) {
			if (bestPointForKeypoint[k] >= 0) addLocalMapMatch(matches, bestPointForKeypoint[k], static_cast<int>(k));
		}
		return matches;
	}

	bool StereoFrontend::trackFromLocalKeyframes()
	{
		// Local map = map points observed by the local keyframes (newest keyframe +
		// its best covisible ones). They are matched to the current left image,
		// then PnP gives the current pose.
		if (localKeyframeIndices_.empty() || currentLeft_.descriptors.empty()) return false;

		const std::vector<std::size_t> localKeyframes = selectLocalKeyframes();
		stats_.localKeyframes = localKeyframes.size();

		std::unordered_set<int> seenMapPointIndices;
		std::vector<int> localMapPointIndices;
		for (const std::size_t keyframeIndex : localKeyframes) {
			if (keyframeIndex >= keyframes_.size()) continue;
			const Keyframe& keyframe = keyframes_.at(keyframeIndex);
			if (!keyframe.active) continue;

			for (const int mapPointIndex : keyframe.mapPointIds) {
				if (mapPointIndex < 0 || static_cast<std::size_t>(mapPointIndex) >= mapPoints_.size()) continue;
				if (!mapPoints_.at(mapPointIndex).valid) continue;
				if (seenMapPointIndices.insert(mapPointIndex).second) {
					localMapPointIndices.push_back(mapPointIndex);
				}
			}
		}

		stats_.localMapPoints = localMapPointIndices.size();
		stats_.localMapVisible = countVisibleMapPoints(localMapPointIndices, predictedWorldFromCamera_);

		// Projection search first; if the prediction was too far off (e.g. turn
		// entry), retry with twice the radius, then fall back to global kNN.
		LocalMapMatches matches;
		if (useProjectionSearch_) {
			matches = matchLocalMapByProjection(localMapPointIndices, predictedWorldFromCamera_,
												projectionRadiusPixels_);
			stats_.matchMethod = "proj";
			if (matches.size() < projectionMinMatches_) {
				matches = matchLocalMapByProjection(localMapPointIndices, predictedWorldFromCamera_,
													2.0F * projectionRadiusPixels_);
				stats_.matchMethod = "proj_wide";
			}
		}
		// Still few matches after the wide search: the prediction is probably off
		// (e.g. right after a kNN-recovered frame), so also try kNN and keep the
		// larger set. A PnP on ~30 wide-window matches once derailed a whole run.
		if (!useProjectionSearch_ || matches.size() < 3 * projectionMinMatches_) {
			LocalMapMatches descriptorMatches = matchLocalMapByDescriptor(localMapPointIndices);
			if (!useProjectionSearch_ || descriptorMatches.size() > matches.size()) {
				matches = std::move(descriptorMatches);
				stats_.matchMethod = "knn";
			}
		}

		const std::vector<cv::Point3f>& objectPoints = matches.objectPoints;
		const std::vector<cv::Point2f>& imagePoints = matches.imagePoints;
		const std::vector<int>& correspondenceMapPointIndices = matches.mapPointIndices;
		const std::vector<int>& correspondenceCurrentKeypointIndices = matches.keypointIndices;

		// PnP. If projection-search matches fail (enough matches, but many wrong
		// because the prediction was off), retry once with global kNN matching.
		// The references above stay valid: they point into `matches`.
		cv::Mat rvec, tvec, inlierIndices;
		bool poseSolved = false;
		for (int attempt = 0; attempt < 2; ++attempt) {
			std::cout << "Local-map 3D -> current-image 2D pairs: " << objectPoints.size()
					  << " (" << stats_.matchMethod << ")\n";
			stats_.localPairs = objectPoints.size();

			if (objectPoints.size() >= 15) {
				poseSolved = cv::solvePnPRansac(
					objectPoints, imagePoints, cameraMatrix_, distortionCoefficients_,
					rvec, tvec, false, 150, 3.0, 0.99, inlierIndices, cv::SOLVEPNP_ITERATIVE);
				stats_.localInliers = static_cast<std::size_t>(inlierIndices.rows);
				if (poseSolved && inlierIndices.rows >= 15) break;
				poseSolved = false;
			}

			const bool canRetry = attempt == 0 && useProjectionSearch_ && stats_.matchMethod != "knn";
			if (!canRetry) break;
			matches = matchLocalMapByDescriptor(localMapPointIndices);
			stats_.matchMethod = "knn_retry";
		}
		if (!poseSolved) {
			std::cout << "Local-map PnP failed. Inliers: " << inlierIndices.rows
						<< " / " << objectPoints.size() << '\n';
			return false;
		}

		cv::Mat rotationMatrix;
		cv::Rodrigues(rvec, rotationMatrix);
		Eigen::Isometry3d cameraFromWorld = Eigen::Isometry3d::Identity();
		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 3; ++column) {
				cameraFromWorld.linear()(row, column) = rotationMatrix.at<double>(row, column);
			}
			cameraFromWorld.translation()(row) = tvec.at<double>(row, 0);
		}
		worldFromCamera_ = cameraFromWorld.inverse();

		std::vector<cv::Point2f> inlierPixels;
		inlierPixels.reserve(static_cast<std::size_t>(inlierIndices.rows));
		for (int row = 0; row < inlierIndices.rows; ++row) {
			const int correspondenceIndex = inlierIndices.at<int>(row, 0);
			const int mapPointIndex = correspondenceMapPointIndices.at(correspondenceIndex);
			const int currentKeypointIndex = correspondenceCurrentKeypointIndices.at(correspondenceIndex);
			currentLeft_.mapPointIds.at(currentKeypointIndex) = mapPointIndex;
			inlierPixels.push_back(imagePoints.at(correspondenceIndex));
		}
		stats_.gridCoverage = computeGridCoverage(inlierPixels);
		stats_.hullAreaRatio = computeHullAreaRatio(inlierPixels);

		// With the pose now known accurately: link more keypoints to existing
		// points (prevents duplicates), then update the culling statistics.
		if (useMapPointFusion_) stats_.associatedAfterPose = associateLocalMapAfterPose(localMapPointIndices);
		if (useMapPointCulling_) updateMapPointStatistics(localMapPointIndices);

		std::cout << "Local-map PnP succeeded. Inliers: " << inlierIndices.rows
					<< " / " << objectPoints.size() << " | +" << stats_.associatedAfterPose
					<< " associated after pose\n";
		return true;
	}

	std::vector<TemporalPixelMatch> StereoFrontend::findTemporalPixelMatches() const
	{
		std::vector<TemporalPixelMatch> matches;
		if (previousLeft_.descriptors.empty() || currentLeft_.descriptors.empty()) return matches;

		std::vector<std::vector<cv::DMatch>> candidates;
		temporalMatcher_.knnMatch(previousLeft_.descriptors, currentLeft_.descriptors, candidates, 2);
		std::unordered_set<int> usedCurrentIndices;

		for (const auto& twoMatches : candidates) {
			if (twoMatches.size() < 2) continue;
			const cv::DMatch& best = twoMatches[0];
			const cv::DMatch& second = twoMatches[1];
			if (best.distance >= 0.85F * second.distance || best.distance > 80.0F) continue;
			if (!usedCurrentIndices.insert(best.trainIdx).second) continue;

			matches.push_back({best.queryIdx, best.trainIdx, best.distance,
								previousLeft_.keypoints.at(best.queryIdx).pt,
								currentLeft_.keypoints.at(best.trainIdx).pt});
		}
		std::cout << "Temporal pixel descriptor matches: " << matches.size() << '\n';
		return matches;
	}

	bool StereoFrontend::trackFromPreviousStereo()
	{
		// No essential-matrix pre-filter: E is degenerate when translation is small
		// (slow driving, stops, rotation-dominated turns) and then throws away good
		// matches. PnP RANSAC on 3D-2D pairs is the outlier filter and has no such
		// degeneracy.
		const std::vector<TemporalPixelMatch> candidates = findTemporalPixelMatches();

		std::unordered_map<int, Eigen::Vector3d> previousPointByLeftIndex;
		for (const StereoMatch& match : previousStereoMatches_) {
			previousPointByLeftIndex.emplace(match.leftKeypointIndex, match.pointLeftCamera);
		}

		std::vector<cv::Point3f> previousObjectPoints;
		std::vector<cv::Point2f> currentImagePoints;
		std::vector<int> previousKeypointIndices;
		std::vector<int> currentKeypointIndices;

		for (const TemporalPixelMatch& match : candidates) {
			const auto pointIterator = previousPointByLeftIndex.find(match.previousLeftKeypointIndex);
			if (pointIterator == previousPointByLeftIndex.end()) continue;
			const Eigen::Vector3d& previousPoint = pointIterator->second;
			if (previousPoint.z() > maximumTemporalTrackingDepthMeters_) continue;

			previousObjectPoints.emplace_back(static_cast<float>(previousPoint.x()),
												static_cast<float>(previousPoint.y()),
												static_cast<float>(previousPoint.z()));
			currentImagePoints.push_back(match.currentPixel);
			previousKeypointIndices.push_back(match.previousLeftKeypointIndex);
			currentKeypointIndices.push_back(match.currentLeftKeypointIndex);
		}

		std::cout << "Previous-stereo 3D -> current-image 2D pairs: "
					<< previousObjectPoints.size() << '\n';
		stats_.fallbackPairs = previousObjectPoints.size();
		if (previousObjectPoints.size() < 8) return false;

		cv::Mat rvec, tvec, inlierIndices;
		const bool poseSolved = cv::solvePnPRansac(
			previousObjectPoints, currentImagePoints, cameraMatrix_, distortionCoefficients_,
			rvec, tvec, false, 150, 3.0, 0.99, inlierIndices, cv::SOLVEPNP_ITERATIVE);
		stats_.fallbackInliers = static_cast<std::size_t>(inlierIndices.rows);
		if (!poseSolved || inlierIndices.rows < 8) {
			std::cout << "Previous-to-current PnP failed. Inliers: " << inlierIndices.rows
						<< " / " << previousObjectPoints.size() << '\n';
			return false;
		}

		cv::Mat rotationMatrix;
		cv::Rodrigues(rvec, rotationMatrix);
		Eigen::Isometry3d currentFromPrevious = Eigen::Isometry3d::Identity();
		for (int row = 0; row < 3; ++row) {
			for (int column = 0; column < 3; ++column) {
				currentFromPrevious.linear()(row, column) = rotationMatrix.at<double>(row, column);
			}
			currentFromPrevious.translation()(row) = tvec.at<double>(row, 0);
		}
		worldFromCamera_ = worldFromCamera_ * currentFromPrevious.inverse();

		// Transfer only confirmed old landmarks to this frame.
		std::vector<cv::Point2f> inlierPixels;
		inlierPixels.reserve(static_cast<std::size_t>(inlierIndices.rows));
		for (int row = 0; row < inlierIndices.rows; ++row) {
			const int correspondenceIndex = inlierIndices.at<int>(row, 0);
			inlierPixels.push_back(currentImagePoints.at(correspondenceIndex));
			const int previousKeypointIndex = previousKeypointIndices.at(correspondenceIndex);
			const int currentKeypointIndex = currentKeypointIndices.at(correspondenceIndex);
			const int mapPointIndex = previousLeft_.mapPointIds.at(previousKeypointIndex);
			if (mapPointIndex >= 0) {
				currentLeft_.mapPointIds.at(currentKeypointIndex) = mapPointIndex;
			}
		}
		stats_.gridCoverage = computeGridCoverage(inlierPixels);
		stats_.hullAreaRatio = computeHullAreaRatio(inlierPixels);

		std::cout << "Previous-to-current PnP succeeded. Inliers: " << inlierIndices.rows
					<< " / " << previousObjectPoints.size() << '\n';
		return true;
	}
}
