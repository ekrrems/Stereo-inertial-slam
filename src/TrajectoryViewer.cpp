#include "TrajectoryViewer.hpp"

#include <Eigen/Eigenvalues>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace RigVis
{
	namespace
	{
		constexpr int imageWidth = 1000;
		constexpr int imageHeight = 700;
		constexpr double covarianceSigma = 2.0;

		cv::Point toPixel(
			const Eigen::Vector2d& point,
			const Eigen::Vector2d& center,
			const double pixelsPerMeter)
		{
			return cv::Point(
				static_cast<int>(
					imageWidth * 0.5 +
					(point.x() - center.x()) * pixelsPerMeter),

				static_cast<int>(
					imageHeight * 0.5 -
					(point.y() - center.y()) * pixelsPerMeter));
		}
	}  // namespace

	void TrajectoryViewer::update(
		const FilterState& state,
		const CovarianceMatrix& covariance)
	{
		const Eigen::Vector2d currentXY(
			state.positionW.x(),
			state.positionW.y());

		pathXY_.push_back(currentXY);

		Eigen::Vector2d minimum = currentXY;
		Eigen::Vector2d maximum = currentXY;

		for (const Eigen::Vector2d& point : pathXY_) {
			minimum = minimum.cwiseMin(point);
			maximum = maximum.cwiseMax(point);
		}

		// Include the world origin in the visible area.
		minimum = minimum.cwiseMin(Eigen::Vector2d::Zero());
		maximum = maximum.cwiseMax(Eigen::Vector2d::Zero());

		const Eigen::Matrix2d covarianceXY =
			0.5 * (
				covariance.block<2, 2>(3, 3) +
				covariance.block<2, 2>(3, 3).transpose());

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
			covarianceXY);

		const Eigen::Vector2d eigenvalues =
			solver.eigenvalues().cwiseMax(0.0);

		const double maxUncertainty =
			covarianceSigma * std::sqrt(eigenvalues.maxCoeff());

		minimum.array() -= maxUncertainty + 1.0;
		maximum.array() += maxUncertainty + 1.0;

		Eigen::Vector2d span = maximum - minimum;

		// Avoid an absurdly zoomed-in view at the beginning.
		span.x() = std::max(span.x(), 10.0);
		span.y() = std::max(span.y(), 10.0);

		const Eigen::Vector2d center =
			0.5 * (minimum + maximum);

		const double pixelsPerMeter =
			0.90 * std::min(
				static_cast<double>(imageWidth) / span.x(),
				static_cast<double>(imageHeight) / span.y());

		cv::Mat canvas(
			imageHeight,
			imageWidth,
			CV_8UC3,
			cv::Scalar(25, 25, 25));

		// Draw world axes.
		const cv::Point origin =
			toPixel(Eigen::Vector2d::Zero(), center, pixelsPerMeter);

		cv::line(
			canvas,
			cv::Point(0, origin.y),
			cv::Point(imageWidth, origin.y),
			cv::Scalar(70, 70, 70),
			1);

		cv::line(
			canvas,
			cv::Point(origin.x, 0),
			cv::Point(origin.x, imageHeight),
			cv::Scalar(70, 70, 70),
			1);

		// Draw the propagated XY path.
		for (std::size_t index = 1; index < pathXY_.size(); ++index) {
			cv::line(
				canvas,
				toPixel(pathXY_[index - 1], center, pixelsPerMeter),
				toPixel(pathXY_[index], center, pixelsPerMeter),
				cv::Scalar(255, 180, 30),
				2,
				cv::LINE_AA);
		}

		// Draw a 2σ uncertainty ellipse in the XY plane.
		const Eigen::Vector2d majorDirection =
			solver.eigenvectors().col(1);

		const double majorAxisMeters =
			covarianceSigma * std::sqrt(eigenvalues(1));

		const double minorAxisMeters =
			covarianceSigma * std::sqrt(eigenvalues(0));

		const double angleDegrees =
			-std::atan2(
				majorDirection.y(),
				majorDirection.x()) *
			180.0 / 3.14159265358979323846;

		cv::ellipse(
			canvas,
			toPixel(currentXY, center, pixelsPerMeter),
			cv::Size(
				static_cast<int>(majorAxisMeters * pixelsPerMeter),
				static_cast<int>(minorAxisMeters * pixelsPerMeter)),
			angleDegrees,
			0.0,
			360.0,
			cv::Scalar(0, 200, 255),
			2,
			cv::LINE_AA);

		// Current estimated location.
		cv::circle(
			canvas,
			toPixel(currentXY, center, pixelsPerMeter),
			5,
			cv::Scalar(0, 70, 255),
			cv::FILLED,
			cv::LINE_AA);

		cv::putText(
			canvas,
			"RigVis IMU-only trajectory | orange ellipse = 2 sigma XY uncertainty",
			cv::Point(20, 35),
			cv::FONT_HERSHEY_SIMPLEX,
			0.55,
			cv::Scalar(230, 230, 230),
			1,
			cv::LINE_AA);
		const double sigmaX =
			std::sqrt(std::max(0.0, covarianceXY(0, 0)));

		const double sigmaY =
			std::sqrt(std::max(0.0, covarianceXY(1, 1)));

		const std::string uncertaintyText =
			"2 sigma XY: " +
			std::to_string(2.0 * sigmaX) +
			" m, " +
			std::to_string(2.0 * sigmaY) +
			" m";

		cv::putText(
			canvas,
			uncertaintyText,
			cv::Point(20, 65),
			cv::FONT_HERSHEY_SIMPLEX,
			0.55,
			cv::Scalar(0, 200, 255),
			1,
			cv::LINE_AA);

		cv::imshow("RigVis trajectory", canvas);
		cv::waitKey(1);
	}

	void TrajectoryViewer::waitUntilClosed() const
	{
		cv::waitKey(0);
	}
}  // namespace RigVis