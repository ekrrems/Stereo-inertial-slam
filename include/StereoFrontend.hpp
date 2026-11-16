/* RigVis stereo frontend: bounded local-keyframe visual odometry. */
#pragma once

#include <cstddef>
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DenseMapper.hpp"
#include "LocalBundleAdjuster.hpp"
#include "types.hpp"

namespace RigVis
{
	struct StereoMatch
	{
		int leftKeypointIndex = -1;
		int rightKeypointIndex = -1;
		float descriptorDistance = 0.0F;
		float disparityPixels = 0.0F;
		Eigen::Vector3d pointLeftCamera = Eigen::Vector3d::Zero();
	};

	struct ImageFeatures
	{
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		// Keypoint index -> mapPoints_ vector index, or -1.
		std::vector<int> mapPointIds;
	};

	// One measurement of a map point in one keyframe: a row block of the
	// observation vector l in bundle adjustment.
	struct Observation
	{
		std::size_t keyframeId = 0;
		int keypointIndex = -1;
		Eigen::Vector2d pixel = Eigen::Vector2d::Zero();  // (u_L, v)
		double uRight = -1.0;  // right-image x; < 0 means mono (no stereo match in that keyframe)
		int octave = 0;        // ORB pyramid level -> sigma = 1.2^octave px, weight = 1/sigma^2

		bool isStereo() const { return uRight >= 0.0; }
	};

	// Reprojection errors with the CURRENT poses and points: Delta l before any
	// bundle adjustment. "Anchor" = the observation from the keyframe that
	// created the point (must be ~0 px, else the projection/pose convention is
	// wrong). "Re-observations" = later keyframes (what local BA will shrink).
	struct ReprojectionGroupStats
	{
		std::size_t count = 0;
		double medianPixels = 0.0;
		double p90Pixels = 0.0;
		double fractionAbove5Pixels = 0.0;
		double fractionChi2Outliers = 0.0;  // v'Pv > 7.815 (stereo) / 5.991 (mono): the BA outlier test
	};

	struct ReprojectionReport
	{
		std::size_t keyframesChecked = 0;
		std::size_t stereoObservations = 0;
		std::size_t monoObservations = 0;
		std::size_t behindCamera = 0;
		ReprojectionGroupStats anchor;
		ReprojectionGroupStats reobserved;
	};

	struct SparseMapPoint
	{
		std::size_t id = 0;
		Eigen::Vector3d positionWorld = Eigen::Vector3d::Zero();
		cv::Mat descriptor;
		std::vector<Observation> observations;
		Eigen::Matrix3d positionCovariance = Eigen::Matrix3d::Identity() * 1e-2;
		bool valid = true;

		// Map point management (culling):
		std::size_t firstKeyframeId = 0;  // keyframe that created the point
		int visibleCount = 0;             // frames where it projected inside the image
		int foundCount = 0;               // frames where it was actually matched

		std::uint8_t intensity = 128;     // gray value at creation (for PLY export)
	};

	struct Keyframe
	{
		std::size_t id = 0;
		bool active = true;
		std::size_t frameIndex = 0;       // dataset frame number (image file name for export)
		double timestamp_s = 0.0;
		Eigen::Isometry3d worldFromCamera = Eigen::Isometry3d::Identity();
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		// Keypoint index -> mapPoints_ vector index, or -1.
		std::vector<int> mapPointIds;
	};

	struct TemporalPixelMatch
	{
		int previousLeftKeypointIndex = -1;
		int currentLeftKeypointIndex = -1;
		float descriptorDistance = 0.0F;
		cv::Point2f previousPixel;
		cv::Point2f currentPixel;
	};

	// Diagnostics for one processed stereo frame. Filled by process(), read by
	// the caller (main writes it to CSV together with the IMU yaw rate).
	struct FrameStats
	{
		std::size_t frameIndex = 0;
		double timestamp_s = 0.0;

		std::size_t features = 0;          // left ORB keypoints
		std::size_t stereoMatches = 0;     // valid left-right stereo points

		// Main tracker: local map of the last N keyframes.
		std::size_t localMapPoints = 0;    // unique points in the local map
		std::size_t localMapVisible = 0;   // of those, inside the image at the PREDICTED pose
		std::string matchMethod;           // proj | proj_wide | knn : how the local-map pairs were found
		std::size_t localKeyframes = 0;    // keyframes whose points formed the local map
		std::size_t localPairs = 0;        // 3D-2D pairs after the ratio test
		std::size_t localInliers = 0;      // PnP RANSAC inliers

		// Fallback tracker: previous-frame stereo 3D -> current 2D.
		std::size_t fallbackPairs = 0;
		std::size_t fallbackInliers = 0;

		std::size_t trackedMapPoints = 0;  // current keypoints associated to map points
		double gridCoverage = 0.0;         // fraction of grid cells with >= 1 pose inlier
		double hullAreaRatio = 0.0;        // convex hull of pose inliers / image area (spread)

		// init | init_fail | local | prev_stereo | reinit | lost | no_desc
		std::string path = "none";
		bool keyframeInserted = false;
		// fallback | max_interval | low_tracked | ratio | close | reinit | init
		std::string keyframeReason;

		// Local bundle adjustment (only on keyframe frames).
		bool bundleAdjustmentRan = false;
		double bundleAdjustmentMilliseconds = 0.0;
		std::size_t bundleAdjustmentOutliers = 0;
		double bundleAdjustmentRmsBefore = 0.0;  // RMS of residuals in sigma units
		double bundleAdjustmentRmsAfter = 0.0;

		// Map point management.
		std::size_t associatedAfterPose = 0;  // extra matches from the tight second search
		std::size_t culledPoints = 0;         // points removed at this keyframe
		double denseMilliseconds = 0.0;       // SGBM dense cloud time (keyframes only)

		// Loop closure (only on keyframe frames).
		long loopCandidate = -1;              // verified old keyframe id, -1 = none
		std::size_t loopInliers = 0;
		bool loopCorrected = false;
		double loopCorrectionMeters = 0.0;    // how far the current pose was moved
	};

	std::string frameStatsCsvHeader();
	std::string frameStatsCsvRow(const FrameStats& stats);

    class StereoFrontend
    {
		public:
			explicit StereoFrontend(const StereoCalibration& calibration);

			bool process(const StereoFrame& frame);

			// Pose of every processed frame, rebuilt from the CURRENT (possibly
			// loop-corrected) keyframe poses: frame pose = reference keyframe pose *
			// stored relative pose. Use this for evaluation; currentPose() at the
			// time of processing never sees later corrections.
			std::vector<Eigen::Isometry3d> finalTrajectory() const;

			const std::vector<SparseMapPoint>& mapPoints() const { return mapPoints_; }
			const std::vector<Keyframe>& keyframes() const { return keyframes_; }
			const Eigen::Isometry3d& currentPose() const { return worldFromCamera_; }
			const ImageFeatures& currentLeftFeatures() const { return currentLeft_; }
			const ImageFeatures& currentRightFeatures() const { return currentRight_; }
			const FrameStats& lastStats() const { return stats_; }

			// Reprojection errors of all observations made by the last
			// windowKeyframes keyframes (0 = whole map), with the current estimates.
			ReprojectionReport reprojectionReport(std::size_t windowKeyframes = 10) const;
			static void printReprojectionReport(const ReprojectionReport& report);

			// Local BA switch, e.g. for A/B runs: frontend.setLocalBundleAdjustment(false).
			void setLocalBundleAdjustment(bool enabled) { enableLocalBundleAdjustment_ = enabled; }

			// Tracking switches, e.g. for A/B runs.
			void setProjectionSearch(bool enabled) { useProjectionSearch_ = enabled; }
			void setCovisibilityLocalMap(bool enabled) { useCovisibilityLocalMap_ = enabled; }
			void setMapPointFusion(bool enabled) { useMapPointFusion_ = enabled; }
			void setMapPointCulling(bool enabled) { useMapPointCulling_ = enabled; }

			// Loop closure: Off, DetectOnly (verify and log, change nothing), Correct.
			enum class LoopClosureMode { Off, DetectOnly, Correct };
			void setLoopClosure(LoopClosureMode mode) { loopClosureMode_ = mode; }
			std::size_t loopsClosed() const { return loopsClosed_; }

			// ---- Map export (src/MapExport.cpp). PLY opens in Metashape
			// (Import -> Points), CloudCompare, MeshLab, Blender. zUp converts the
			// camera-style world frame (x right, y down, z forward) to z-up.
			bool exportSparseMapPly(const std::string& path, bool zUp = true) const;
			// Dense cloud colors: Rgb = from the color camera (KITTI image_02),
			// Height = Turbo colormap over height, Gray = left image intensity,
			// Auto = Rgb if color images were available, otherwise Height.
			enum class DenseColorMode { Auto, Rgb, Height, Gray };
			bool exportDenseMapPly(const std::string& path, double voxelSizeMeters = 0.1, bool zUp = true,
								   DenseColorMode colorMode = DenseColorMode::Auto) const;
			bool exportCamerasPly(const std::string& path, bool zUp = true) const;
			// Per keyframe: image file name, position, orientation (Metashape reference import).
			bool exportKeyframesCsv(const std::string& path, bool zUp = true) const;
			void setDenseMapping(bool enabled) { denseOptions_.enabled = enabled; }

			// Covisibility graph: keyframes sharing the most map points with keyframeId,
			// strongest first, only edges with at least minWeight shared points.
			std::vector<std::size_t> bestCovisibleKeyframes(std::size_t keyframeId, std::size_t count,
															 int minWeight) const;
			const BundleAdjustmentSummary& lastBundleAdjustment() const { return lastBundleAdjustment_; }

		private:
			// Diagnostics helpers.
			std::size_t countVisibleMapPoints(const std::vector<int>& mapPointIndices,
											  const Eigen::Isometry3d& worldFromCamera) const;
			double computeGridCoverage(const std::vector<cv::Point2f>& pixels) const;
			double computeHullAreaRatio(const std::vector<cv::Point2f>& pixels) const;

			// Feature extraction with an even spatial spread.
			void extractCurrentFeatures(const StereoFrame& frame);
			void detectDistributed(const cv::Ptr<cv::ORB>& orb, const cv::Mat& image,
								   ImageFeatures& features) const;
			std::vector<cv::KeyPoint> distributeOnGrid(std::vector<cv::KeyPoint>& candidates,
													   int width, int height) const;

			// Rectified stereo: search only the same row and the valid disparity range.
			std::size_t findCurrentStereoMatches();
			std::size_t addNewMapPointsFromCurrentStereo();

			// Returns why a keyframe should be inserted now, or nullptr.
			const char* keyframeReason(std::size_t trackedMapPoints, bool usedFallback) const;
			// Runs local BA right after a keyframe was saved, updates the current
			// pose, and writes a before/after debug image every few keyframes.
			void runLocalBundleAdjustmentOnNewKeyframe(const StereoFrame& frame);

			// For the debug image: reprojection of the newest keyframe's re-observations.
			struct ObservationProjection
			{
				std::size_t pointIndex = 0;
				int keypointIndex = -1;
				cv::Point2f measured;
				cv::Point2f predicted;
			};
			std::vector<ObservationProjection> projectNewestKeyframeReobservations() const;
			void saveBundleAdjustmentDebugImage(const cv::Mat& image,
												const std::vector<ObservationProjection>& before,
												const std::vector<ObservationProjection>& after) const;

			void saveCurrentAsKeyframe(const StereoFrame& frame,
										std::size_t addedPointCount);
			void assignNewToPrevious(const StereoFrame& frame);

			// Normal bounded-cost tracker. It only uses landmarks observed by the
			// latest localTrackingKeyframeCount_ keyframes.
			bool trackFromLocalKeyframes();

			// 3D map point <-> 2D current keypoint pairs for PnP.
			struct LocalMapMatches
			{
				std::vector<cv::Point3f> objectPoints;
				std::vector<cv::Point2f> imagePoints;
				std::vector<int> mapPointIndices;
				std::vector<int> keypointIndices;
				std::size_t size() const { return objectPoints.size(); }
			};
			// Keyframes whose points form the tracking local map.
			std::vector<std::size_t> selectLocalKeyframes() const;
			// Global descriptor matching (kNN + ratio test against ALL current keypoints).
			LocalMapMatches matchLocalMapByDescriptor(const std::vector<int>& localMapPointIndices) const;
			// Projection search: predict where each point appears, match only near there.
			LocalMapMatches matchLocalMapByProjection(const std::vector<int>& localMapPointIndices,
													  const Eigen::Isometry3d& predictedWorldFromCamera,
													  float radiusPixels) const;
			void addLocalMapMatch(LocalMapMatches& matches, int mapPointIndex, int keypointIndex) const;

			// Adds the new keyframe's edges to the covisibility graph.
			void updateCovisibility(std::size_t keyframeId);

			// Fusion: after the pose is solved, a tight projection search links more
			// keypoints to EXISTING points, so no duplicates get created at keyframes.
			std::size_t associateLocalMapAfterPose(const std::vector<int>& localMapPointIndices);
			// Culling statistics: visible (projects into the image) vs found (matched).
			void updateMapPointStatistics(const std::vector<int>& localMapPointIndices);
			// Removes recently created points that are rarely found again.
			std::size_t cullRecentMapPoints();

			// The actual per-frame pipeline; process() calls it and records the frame.
			bool processFrame(const StereoFrame& frame);

			struct FrameRecord
			{
				long referenceKeyframe = -1;  // -1: before initialisation
				Eigen::Isometry3d keyframeFromCamera = Eigen::Isometry3d::Identity();
			};
			std::vector<FrameRecord> frameRecords_;

			// ---- Loop closure (src/LoopClosing.cpp)
			struct LoopVerification
			{
				bool accepted = false;
				std::size_t candidateKeyframe = 0;
				std::size_t inliers = 0;
				std::size_t matches = 0;
				Eigen::Isometry3d worldFromCamera = Eigen::Isometry3d::Identity();  // current pose in the OLD map
				std::vector<std::pair<int, int>> inlierPairs;  // (current keypoint, old map point)
			};
			// Called after each new keyframe: detect -> verify -> (correct).
			void detectAndCloseLoop();
			std::vector<std::size_t> findLoopCandidates() const;
			LoopVerification verifyLoopCandidate(std::size_t candidateKeyframe) const;
			void correctLoop(const LoopVerification& loop);
			// Merge duplicates: current keypoints matched to old points take the old point.
			void fuseLoopMapPoints(const LoopVerification& loop);

			// Frame-to-frame metric fallback: previous stereo 3D -> current 2D.
			bool trackFromPreviousStereo();

			std::vector<TemporalPixelMatch> findTemporalPixelMatches() const;

			StereoCalibration calibration_;
			StereoFrame previousFrame_;

			cv::Mat cameraMatrix_;
			cv::Mat currentLeftImage_;  // current frame's left image (point intensities)

			// ---- Dense map: one cloud per Nth keyframe, in that keyframe's camera frame
			DenseCloudOptions denseOptions_;
			std::unordered_map<std::size_t, DenseCloud> denseClouds_;
			cv::Mat distortionCoefficients_;

			bool frontendInitialized_ = false;
			bool mapInitialized_ = false;
			bool keyframeDetected_ = false;

			ImageFeatures currentLeft_;
			ImageFeatures currentRight_;
			ImageFeatures previousLeft_;
			ImageFeatures previousRight_;

			// ---- Feature extraction
			int targetFeatures_ = 1500;        // kept per image after grid distribution
			int candidateMultiplier_ = 4;      // detect this many times more, then distribute
			int fastThreshold_ = 7;            // low, so weak-texture cells still get corners
			int detectionGridRows_ = 8;
			int detectionGridCols_ = 24;

			cv::Ptr<cv::ORB> orbLeft_;
			cv::Ptr<cv::ORB> orbRight_;
			cv::BFMatcher temporalMatcher_{cv::NORM_HAMMING};

			// ---- Stereo matching
			float maximumEpipolarErrorPixels_ = 2.0F;  // at octave 0, scaled by 1.2^octave
			float minimumDisparityPixels_ = 1.0F;
			double minimumDepthMeters_ = 0.5;
			double maximumDepthMeters_ = 80.0;
			double maximumTemporalTrackingDepthMeters_ = 35.0;
			int maximumStereoDescriptorDistance_ = 75; // Hamming, ORB-SLAM's (TH_HIGH+TH_LOW)/2

			// ---- Keyframe policy (ORB-SLAM2 NeedNewKeyFrame, simplified)
			std::size_t framesSinceLastKeyframe_ = 0;
			std::size_t minKeyframeIntervalFrames_ = 1;
			std::size_t maxKeyframeIntervalFrames_ = 10;   // 1 s at 10 Hz
			// ORB-SLAM2 uses 0.75, but it also tracks via projection search and has a
			// mapping thread. Tuned on KITTI 0027 frames 140-300: 0.75 -> KF every
			// ~2.3 frames, 0.6 -> every ~3 frames, same trajectory, higher min inliers.
			double keyframeTrackedRatio_ = 0.6;
			std::size_t minTrackedForKeyframe_ = 50;        // absolute floor
			// Close-point rule. Off: with descriptor-only tracking most close stereo
			// points stay untracked, so it fired on ~80% of frames. Re-enable once
			// tracking uses projection search (motion model + frustum culling).
			bool useCloseKeyframeRule_ = false;
			std::size_t closeTrackedMax_ = 100;
			std::size_t closeUntrackedMin_ = 70;
			double closeDepthMeters_ = 0.0;                 // 35 * baseline, set in constructor
			std::size_t referenceTracked_ = 0;              // tracked count on the first frame after the last keyframe

			std::vector<Keyframe> keyframes_;
			std::size_t nextKeyframeId_ = 0;

			// These are vector indices, not Keyframe::id values.
			std::deque<std::size_t> localKeyframeIndices_;
			std::size_t localTrackingKeyframeCount_ = 5;

			// ---- Covisibility graph: keyframe id -> (other keyframe id -> shared map points)
			std::vector<std::unordered_map<std::size_t, int>> covisibilityWeights_;
			bool useCovisibilityLocalMap_ = true;
			int covisibilityMinWeight_ = 15;              // ORB-SLAM's edge threshold

			// ---- Projection search (tracking against the local map)
			bool useProjectionSearch_ = true;
			float projectionRadiusPixels_ = 15.0F;        // retried at 2x if too few matches
			std::size_t projectionMinMatches_ = 30;       // below this: wider radius, then kNN fallback
			int projectionMaxDescriptorDistance_ = 75;    // Hamming
			float projectionRatio_ = 0.8F;                // best < ratio * second-best inside the window
			int projectionCellPixels_ = 20;               // keypoint lookup grid

			// ---- Map point management (ORB-SLAM style)
			bool useMapPointFusion_ = true;
			float fusionRadiusPixels_ = 4.0F;             // second search with the SOLVED pose
			double fusionMaxDepthRatioError_ = 0.15;      // stereo depth must agree within 15 %
			bool useMapPointCulling_ = true;
			double cullMinFoundRatio_ = 0.25;             // found / visible
			int cullMinVisibleForRatio_ = 4;              // only judge the ratio after this many sightings
			std::size_t cullMinObservations_ = 2;         // keyframes that must have seen the point ...
			std::size_t cullAfterKeyframes_ = 2;          // ... this many keyframes after its creation
			std::vector<std::size_t> recentMapPoints_;    // points still on probation

			// ---- Loop closure
			LoopClosureMode loopClosureMode_ = LoopClosureMode::Correct;
			double loopSearchRadiusMeters_ = 50.0;        // candidates: old keyframes this close (must exceed drift)
			double loopMaxViewAngleDegrees_ = 45.0;       // ... looking in a similar direction
			std::size_t loopMinKeyframeGap_ = 100;        // ... and at least this many keyframes older
			std::size_t loopMaxCandidates_ = 3;
			std::size_t loopMinInliers_ = 60;             // geometric verification thresholds
			double loopMinInlierRatio_ = 0.3;
			std::size_t loopCooldownKeyframes_ = 30;      // no new loop right after a correction
			bool globalBundleAdjustmentAfterLoop_ = true;
			// Consistency: a loop is corrected only if the previous keyframe also
			// verified a loop to the same place (one lucky match is not enough).
			long pendingLoopCandidate_ = -1;
			std::size_t pendingLoopKeyframe_ = 0;
			std::size_t lastLoopKeyframe_ = 0;
			std::size_t loopsClosed_ = 0;

			std::vector<SparseMapPoint> mapPoints_;
			std::vector<StereoMatch> currentStereoMatches_;
			std::vector<StereoMatch> previousStereoMatches_;
			std::size_t nextMapPointId_ = 0;

			// T_wc: camera-coordinate point -> world-coordinate point.
			Eigen::Isometry3d worldFromCamera_ = Eigen::Isometry3d::Identity();

			// ---- Constant-velocity motion model
			// velocity_ = T_{prev,cur}: the last frame-to-frame motion, assumed to repeat.
			Eigen::Isometry3d velocity_ = Eigen::Isometry3d::Identity();
			Eigen::Isometry3d predictedWorldFromCamera_ = Eigen::Isometry3d::Identity();
			std::size_t minStereoForReinit_ = 30;

			// Print a reprojection report every N keyframes (0 = off).
			std::size_t reprojectionReportEveryKeyframes_ = 50;

			// ---- Local bundle adjustment
			bool enableLocalBundleAdjustment_ = true;
			BundleAdjustmentOptions localBundleAdjustmentOptions_;
			BundleAdjustmentSummary lastBundleAdjustment_;
			std::size_t bundleAdjustmentDebugEveryKeyframes_ = 20;  // 0 = no debug images
			std::string bundleAdjustmentDebugDirectory_ = "ba_debug";

			// Diagnostics state.
			FrameStats stats_;
			std::size_t frameIndex_ = 0;
			int imageWidth_ = 0;
			int imageHeight_ = 0;
			int gridRows_ = 6;
			int gridCols_ = 20;
	};
}
