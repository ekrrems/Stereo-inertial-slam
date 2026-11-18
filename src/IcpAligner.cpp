#include "IcpAligner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include <Eigen/Eigenvalues>

#include "so3.hpp"

namespace RigVis
{
	namespace
	{
		// Sparse voxel grid: integer cell -> point indices.
		struct GridKey
		{
			std::int64_t operator()(const Eigen::Vector3i& c) const
			{
				return ((static_cast<std::int64_t>(c.x()) & 0x1FFFFF) << 42) |
					   ((static_cast<std::int64_t>(c.y()) & 0x1FFFFF) << 21) | (static_cast<std::int64_t>(c.z()) & 0x1FFFFF);
			}
		};

		Eigen::Vector3i cellOf(const Eigen::Vector3d& p, double inverseSize)
		{
			return Eigen::Vector3i(static_cast<int>(std::floor(p.x() * inverseSize)),
								   static_cast<int>(std::floor(p.y() * inverseSize)),
								   static_cast<int>(std::floor(p.z() * inverseSize)));
		}

		std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3f>& points, double voxel, double maxRange)
		{
			struct Acc { Eigen::Vector3d sum = Eigen::Vector3d::Zero(); int n = 0; };
			std::unordered_map<std::int64_t, Acc> cells;
			cells.reserve(points.size() / 2);
			const double inv = 1.0 / voxel;
			for (const Eigen::Vector3f& pf : points) {
				if (pf.z() > maxRange) continue;
				const Eigen::Vector3d p = pf.cast<double>();
				Acc& a = cells[GridKey()(cellOf(p, inv))];
				a.sum += p;
				++a.n;
			}
			std::vector<Eigen::Vector3d> out;
			out.reserve(cells.size());
			for (const auto& [k, a] : cells) out.push_back(a.sum / a.n);
			return out;
		}

		class Grid
		{
		public:
			Grid(const std::vector<Eigen::Vector3d>& points, double cellSize)
				: points_(points), inverse_(1.0 / cellSize)
			{
				cells_.reserve(points.size());
				for (std::size_t i = 0; i < points.size(); ++i) cells_[GridKey()(cellOf(points[i], inverse_))].push_back(i);
			}

			// Nearest point within maxDistance (searches the 27 surrounding cells,
			// valid while maxDistance <= cell size). Returns -1 if none.
			int nearest(const Eigen::Vector3d& p, double maxDistance) const
			{
				const Eigen::Vector3i c = cellOf(p, inverse_);
				double best = maxDistance * maxDistance;
				int bestIndex = -1;
				for (int dx = -1; dx <= 1; ++dx)
					for (int dy = -1; dy <= 1; ++dy)
						for (int dz = -1; dz <= 1; ++dz) {
							const auto it = cells_.find(GridKey()(c + Eigen::Vector3i(dx, dy, dz)));
							if (it == cells_.end()) continue;
							for (const std::size_t i : it->second) {
								const double d = (points_[i] - p).squaredNorm();
								if (d < best) { best = d; bestIndex = static_cast<int>(i); }
							}
						}
				return bestIndex;
			}

			template <typename F>
			void forNeighbours(const Eigen::Vector3d& p, double radius, F&& f) const
			{
				const Eigen::Vector3i c = cellOf(p, inverse_);
				const double r2 = radius * radius;
				for (int dx = -1; dx <= 1; ++dx)
					for (int dy = -1; dy <= 1; ++dy)
						for (int dz = -1; dz <= 1; ++dz) {
							const auto it = cells_.find(GridKey()(c + Eigen::Vector3i(dx, dy, dz)));
							if (it == cells_.end()) continue;
							for (const std::size_t i : it->second)
								if ((points_[i] - p).squaredNorm() <= r2) f(i);
						}
			}

		private:
			const std::vector<Eigen::Vector3d>& points_;
			double inverse_;
			std::unordered_map<std::int64_t, std::vector<std::size_t>> cells_;
		};
	}

	IcpResult alignPointToPlane(const std::vector<Eigen::Vector3f>& targetRaw, const std::vector<Eigen::Vector3f>& sourceRaw,
								const Eigen::Isometry3d& initialTargetFromSource, const IcpOptions& options)
	{
		const auto start = std::chrono::steady_clock::now();
		IcpResult result;
		auto finish = [&](const char* reason) {
			result.reason = reason ? reason : "";
			result.valid = reason == nullptr;
			result.milliseconds =
				std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			return result;
		};

		// ---- 1. Downsample, target normals from the local covariance (PCA).
		const std::vector<Eigen::Vector3d> target = voxelDownsample(targetRaw, options.voxelSize, options.maxRange);
		const std::vector<Eigen::Vector3d> source = voxelDownsample(sourceRaw, options.voxelSize, options.maxRange);
		result.targetPoints = target.size();
		result.sourcePoints = source.size();
		if (target.size() < options.minInliers || source.size() < options.minInliers) return finish("too few points");

		const Grid normalGrid(target, options.normalRadius);
		std::vector<Eigen::Vector3d> normals(target.size(), Eigen::Vector3d::Zero());
		std::vector<bool> hasNormal(target.size(), false);
		for (std::size_t i = 0; i < target.size(); ++i) {
			Eigen::Vector3d mean = Eigen::Vector3d::Zero();
			Eigen::Matrix3d second = Eigen::Matrix3d::Zero();
			std::size_t n = 0;
			normalGrid.forNeighbours(target[i], options.normalRadius, [&](std::size_t j) {
				mean += target[j];
				second += target[j] * target[j].transpose();
				++n;
			});
			if (n < options.minNormalNeighbours) continue;
			mean /= static_cast<double>(n);
			const Eigen::Matrix3d covariance = second / static_cast<double>(n) - mean * mean.transpose();
			const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);  // ascending eigenvalues
			if (eig.eigenvalues()(0) > options.maxPlanarity * eig.eigenvalues()(1)) continue;  // not a surface
			normals[i] = eig.eigenvectors().col(0);
			hasNormal[i] = true;
		}

		// ---- 2. Gauss-Newton on the point-to-plane distances.
		//   T <- Exp(delta) * T,  delta = (omega, v) in the TARGET frame
		//   r_i = n . (T p - q),   dr/domega = (T p) x n,   dr/dv = n
		const Grid searchGrid(target, options.maxCorrespondenceStart);
		Eigen::Isometry3d T = initialTargetFromSource;
		Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
		double sumSquares = 0.0;

		for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
			const double t = options.maxIterations > 1 ? static_cast<double>(iteration) / (options.maxIterations - 1) : 1.0;
			const double maxDistance =
				options.maxCorrespondenceStart + (options.maxCorrespondenceEnd - options.maxCorrespondenceStart) * t;

			H.setZero();
			Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
			std::size_t inliers = 0;
			sumSquares = 0.0;
			for (const Eigen::Vector3d& ps : source) {
				const Eigen::Vector3d p = T * ps;
				const int j = searchGrid.nearest(p, maxDistance);
				if (j < 0 || !hasNormal[static_cast<std::size_t>(j)]) continue;
				const Eigen::Vector3d& n = normals[static_cast<std::size_t>(j)];
				const double r = n.dot(p - target[static_cast<std::size_t>(j)]);
				double w = std::abs(r) <= options.huberDelta ? 1.0 : options.huberDelta / std::abs(r);  // Huber (IRLS)
				if (options.depthWeightScale > 0.0) {
					const double z = ps.z() / options.depthWeightScale;
					w /= 1.0 + z * z * z * z;
				}
				Eigen::Matrix<double, 6, 1> J;
				J.head<3>() = p.cross(n);
				J.tail<3>() = n;
				H.noalias() += w * J * J.transpose();
				b.noalias() += w * J * r;
				sumSquares += r * r;
				++inliers;
			}
			result.inliers = inliers;
			result.iterations = iteration + 1;
			if (inliers < options.minInliers) return finish("too few correspondences");

			const Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(-b);
			if (!delta.allFinite()) return finish("singular system");
			Eigen::Isometry3d update = Eigen::Isometry3d::Identity();
			update.linear() = so3Exp(delta.head<3>()).toRotationMatrix();
			update.translation() = delta.tail<3>();
			T = update * T;
			if (delta.norm() < options.convergenceStep) break;
		}

		// ---- 3. Quality and the measurement's information matrix.
		result.targetFromSource = T;
		result.overlap = static_cast<double>(result.inliers) / static_cast<double>(source.size());
		result.rmse = std::sqrt(sumSquares / std::max<std::size_t>(1, result.inliers));
		const Eigen::Isometry3d correction = T * initialTargetFromSource.inverse();
		result.correctionMeters = correction.translation().norm();
		result.correctionDegrees = Eigen::AngleAxisd(correction.linear()).angle() * 180.0 / EIGEN_PI;

		if (result.overlap < options.minOverlap) return finish("low overlap");
		if (result.rmse > options.maxRmse) return finish("high rmse");
		if (result.correctionMeters > options.maxCorrectionMeters || result.correctionDegrees > options.maxCorrectionDegrees)
			return finish("correction too large");

		// Information = H / sigma^2, scaled down for correlated residuals. Its
		// eigenvalues show which directions the geometry constrains.
		const double sigma2 = std::max(result.rmse * result.rmse, 1e-6);
		result.information = H / sigma2 * (options.effectiveIndependentPoints / static_cast<double>(result.inliers));
		result.information += 1e-9 * Eigen::Matrix<double, 6, 6>::Identity();
		// Condition check on a unit-free version (rotation rows scaled by a typical 10 m lever arm).
		Eigen::Matrix<double, 6, 6> scaled = result.information;
		scaled.topRows<3>() /= 10.0;
		scaled.leftCols<3>() /= 10.0;
		const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(scaled);
		if (eig.eigenvalues()(0) < options.minConditionRatio * eig.eigenvalues()(5)) return finish("degenerate geometry");
		return finish(nullptr);
	}
}
