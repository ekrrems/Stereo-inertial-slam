#include "MapViewer.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace RigVis
{
    MapViewer::MapViewer()
    {
        cv::namedWindow(
            "RigVis sparse map",
            cv::WINDOW_NORMAL);

        cv::namedWindow(
            "RigVis left features",
            cv::WINDOW_NORMAL);

        cv::namedWindow(
            "RigVis right features",
            cv::WINDOW_NORMAL);
    }

    cv::Point MapViewer::projectTopDown(
        const Eigen::Vector3d& pointWorld,
        const double minimumX,
        const double maximumX,
        const double minimumZ,
        const double maximumZ,
        const double pixelsPerMeter,
        const int canvasHeight) const
    {
        const double unusedMaximumX = maximumX;
        const double unusedMaximumZ = maximumZ;

        (void) unusedMaximumX;
        (void) unusedMaximumZ;

        const int pixelX =
            borderPixels_ +
            static_cast<int>(
                (pointWorld.x() - minimumX) *
                pixelsPerMeter);

        // Positive world Z appears upward on screen.
        const int pixelY =
            canvasHeight -
            borderPixels_ -
            static_cast<int>(
                (pointWorld.z() - minimumZ) *
                pixelsPerMeter);

        return cv::Point(pixelX, pixelY);
    }

    void MapViewer::update(
        const StereoFrame& frame,
        const ImageFeatures& leftFeatures,
        const ImageFeatures& rightFeatures,
        const std::vector<SparseMapPoint>& mapPoints,
        const std::vector<Keyframe>& keyframes,
        const Eigen::Isometry3d& worldFromCamera)
    {
        // ---------- Stereo feature windows ----------

        cv::Mat leftDisplay;
        cv::Mat rightDisplay;

        cv::drawKeypoints(
            frame.left.image,
            leftFeatures.keypoints,
            leftDisplay,
            cv::Scalar(0, 255, 0),
            cv::DrawMatchesFlags::DEFAULT);

        cv::drawKeypoints(
            frame.right.image,
            rightFeatures.keypoints,
            rightDisplay,
            cv::Scalar(0, 255, 0),
            cv::DrawMatchesFlags::DEFAULT);

        cv::imshow(
            "RigVis left features",
            leftDisplay);

        cv::imshow(
            "RigVis right features",
            rightDisplay);

        // ---------- Find map extents ----------

        const Eigen::Vector3d currentCameraPosition =
            worldFromCamera.translation();

        double minimumX =
            currentCameraPosition.x();

        double maximumX =
            currentCameraPosition.x();

        double minimumZ =
            currentCameraPosition.z();

        double maximumZ =
            currentCameraPosition.z();

        auto expandBounds =
            [&](const Eigen::Vector3d& point)
            {
                minimumX = std::min(
                    minimumX,
                    point.x());

                maximumX = std::max(
                    maximumX,
                    point.x());

                minimumZ = std::min(
                    minimumZ,
                    point.z());

                maximumZ = std::max(
                    maximumZ,
                    point.z());
            };

        for (const SparseMapPoint& mapPoint : mapPoints) {
            if (!mapPoint.valid) {
                continue;
            }

            expandBounds(
                mapPoint.positionWorld);
        }

        for (const Keyframe& keyframe : keyframes) {
            expandBounds(
                keyframe.worldFromCamera.translation());
        }

        constexpr double paddingMeters = 3.0;

        minimumX -= paddingMeters;
        maximumX += paddingMeters;
        minimumZ -= paddingMeters;
        maximumZ += paddingMeters;

        const double widthMeters =
            std::max(
                maximumX - minimumX,
                1.0);

        const double heightMeters =
            std::max(
                maximumZ - minimumZ,
                1.0);

        const double availableWidthPixels =
            mapWidthPixels_ - 2.0 * borderPixels_;

        const double availableHeightPixels =
            mapHeightPixels_ - 2.0 * borderPixels_;

        const double pixelsPerMeter =
            std::min(
                availableWidthPixels / widthMeters,
                availableHeightPixels / heightMeters);

        cv::Mat mapCanvas(
            mapHeightPixels_,
            mapWidthPixels_,
            CV_8UC3,
            cv::Scalar(25, 25, 25));

        // ---------- Sparse 3D landmarks ----------

        const std::size_t pointStride =
            std::max<std::size_t>(
                1,
                mapPoints.size() / 30000);

        for (std::size_t index = 0;
             index < mapPoints.size();
             index += pointStride)
        {
            const SparseMapPoint& mapPoint =
                mapPoints.at(index);

            if (!mapPoint.valid) {
                continue;
            }

            const cv::Point pixel =
                projectTopDown(
                    mapPoint.positionWorld,
                    minimumX,
                    maximumX,
                    minimumZ,
                    maximumZ,
                    pixelsPerMeter,
                    mapHeightPixels_);

            cv::circle(
                mapCanvas,
                pixel,
                1,
                cv::Scalar(255, 170, 60),
                cv::FILLED);
        }

        // ---------- Keyframe positions and headings ----------

        for (const Keyframe& keyframe : keyframes) {
            const Eigen::Vector3d keyframePosition =
                keyframe.worldFromCamera.translation();

            // OpenCV camera convention: local +Z is forward.
            const Eigen::Vector3d forwardWorld =
                keyframe.worldFromCamera.linear() *
                Eigen::Vector3d(0.0, 0.0, 1.0);

            const Eigen::Vector3d forwardTip =
                keyframePosition +
                1.0 * forwardWorld;

            const cv::Point positionPixel =
                projectTopDown(
                    keyframePosition,
                    minimumX,
                    maximumX,
                    minimumZ,
                    maximumZ,
                    pixelsPerMeter,
                    mapHeightPixels_);

            const cv::Point forwardPixel =
                projectTopDown(
                    forwardTip,
                    minimumX,
                    maximumX,
                    minimumZ,
                    maximumZ,
                    pixelsPerMeter,
                    mapHeightPixels_);

            cv::circle(
                mapCanvas,
                positionPixel,
                3,
                cv::Scalar(0, 255, 0),
                cv::FILLED);

            cv::line(
                mapCanvas,
                positionPixel,
                forwardPixel,
                cv::Scalar(0, 180, 0),
                1);
        }

        // ---------- Current live camera pose ----------

        const Eigen::Vector3d currentForwardWorld =
            worldFromCamera.linear() *
            Eigen::Vector3d(0.0, 0.0, 1.0);

        const Eigen::Vector3d currentForwardTip =
            currentCameraPosition +
            1.5 * currentForwardWorld;

        const cv::Point currentPositionPixel =
            projectTopDown(
                currentCameraPosition,
                minimumX,
                maximumX,
                minimumZ,
                maximumZ,
                pixelsPerMeter,
                mapHeightPixels_);

        const cv::Point currentForwardPixel =
            projectTopDown(
                currentForwardTip,
                minimumX,
                maximumX,
                minimumZ,
                maximumZ,
                pixelsPerMeter,
                mapHeightPixels_);

        cv::circle(
            mapCanvas,
            currentPositionPixel,
            6,
            cv::Scalar(0, 0, 255),
            cv::FILLED);

        cv::line(
            mapCanvas,
            currentPositionPixel,
            currentForwardPixel,
            cv::Scalar(0, 0, 255),
            2);

        // ---------- Labels ----------

        cv::putText(
            mapCanvas,
            "RigVis sparse map: orange = landmarks, green = keyframes, red = live pose",
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(230, 230, 230),
            1);

        cv::putText(
            mapCanvas,
            "Map points: " +
            std::to_string(mapPoints.size()) +
            " | Keyframes: " +
            std::to_string(keyframes.size()),
            cv::Point(20, 55),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(230, 230, 230),
            1);

        cv::imshow(
            "RigVis sparse map",
            mapCanvas);

        cv::waitKey(1);
    }
} // namespace RigVis
