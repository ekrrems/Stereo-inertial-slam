#pragma once

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>

#include "StereoFrontend.hpp"
#include "types.hpp"

namespace RigVis
{
    class MapViewer
    {
    public:
        MapViewer();

        void update(
            const StereoFrame& frame,
            const ImageFeatures& leftFeatures,
            const ImageFeatures& rightFeatures,
            const std::vector<SparseMapPoint>& mapPoints,
            const std::vector<Keyframe>& keyframes,
            const Eigen::Isometry3d& worldFromCamera);

    private:
        cv::Point projectTopDown(
            const Eigen::Vector3d& pointWorld,
            double minimumX,
            double maximumX,
            double minimumZ,
            double maximumZ,
            double pixelsPerMeter,
            int canvasHeight) const;

        int mapWidthPixels_ = 1100;
        int mapHeightPixels_ = 800;
        int borderPixels_ = 50;
    };
} // namespace RigVis