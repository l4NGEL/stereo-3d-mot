#include "s3m/depth/depth_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>

namespace s3m {

std::string DepthMetrics::toString() const {
    std::ostringstream os;
    os << *this;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const DepthMetrics& m) {
    const std::ios_base::fmtflags flags = os.flags();
    const std::streamsize precision = os.precision();

    os << std::fixed << std::setprecision(4);
    os << "RMSE=" << m.rmse << "  MAE=" << m.mae << "  absRel=" << m.abs_rel
       << "  density=" << m.density << " (" << m.evaluated_pixels << "/" << m.gt_valid_pixels
       << ")";
    for (std::size_t i = 0; i < m.bad_thresholds.size() && i < m.bad_fraction.size(); ++i) {
        os << "  bad(" << m.bad_thresholds[i] << ")=" << std::setprecision(2)
           << (100.0 * m.bad_fraction[i]) << "%" << std::setprecision(4);
    }
    os << "  delta<1.25: " << m.delta1 << " / " << m.delta2 << " / " << m.delta3;

    os.flags(flags);
    os.precision(precision);
    return os;
}

DepthMetrics evaluate(const cv::Mat& estimate, const cv::Mat& ground_truth, double valid_min,
                      double valid_max, const std::vector<double>& bad_thresholds) {
    CV_Assert(estimate.size() == ground_truth.size());
    CV_Assert(estimate.channels() == 1 && ground_truth.channels() == 1);

    cv::Mat est;
    cv::Mat gt;
    estimate.convertTo(est, CV_64F);
    ground_truth.convertTo(gt, CV_64F);

    DepthMetrics m;
    m.bad_thresholds = bad_thresholds;
    m.bad_fraction.assign(bad_thresholds.size(), 0.0);
    std::vector<std::int64_t> bad_counts(bad_thresholds.size(), 0);

    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double sum_abs_rel = 0.0;
    double sum_sq_rel = 0.0;
    std::int64_t evaluated = 0;
    std::int64_t gt_valid = 0;
    std::int64_t d1 = 0;
    std::int64_t d2 = 0;
    std::int64_t d3 = 0;

    for (int y = 0; y < gt.rows; ++y) {
        const double* pe = est.ptr<double>(y);
        const double* pg = gt.ptr<double>(y);
        for (int x = 0; x < gt.cols; ++x) {
            const double g = pg[x];
            const bool gt_ok = std::isfinite(g) && g >= valid_min && g <= valid_max;
            if (gt_ok)
                ++gt_valid;

            const double e = pe[x];
            const bool est_ok = std::isfinite(e) && e >= valid_min && e <= valid_max;
            if (!gt_ok || !est_ok)
                continue;

            const double err = e - g;
            const double abs_err = std::abs(err);
            sum_sq += err * err;
            sum_abs += abs_err;
            sum_abs_rel += abs_err / g;
            sum_sq_rel += (err * err) / g;

            for (std::size_t k = 0; k < bad_thresholds.size(); ++k) {
                if (abs_err > bad_thresholds[k])
                    ++bad_counts[k];
            }

            const double ratio = std::max(e / g, g / e);
            if (ratio < 1.25)
                ++d1;
            if (ratio < 1.25 * 1.25)
                ++d2;
            if (ratio < 1.25 * 1.25 * 1.25)
                ++d3;

            ++evaluated;
        }
    }

    m.evaluated_pixels = static_cast<int>(evaluated);
    m.gt_valid_pixels = static_cast<int>(gt_valid);

    if (evaluated > 0) {
        const double n = static_cast<double>(evaluated);
        m.rmse = std::sqrt(sum_sq / n);
        m.mae = sum_abs / n;
        m.abs_rel = sum_abs_rel / n;
        m.sq_rel = sum_sq_rel / n;
        m.delta1 = static_cast<double>(d1) / n;
        m.delta2 = static_cast<double>(d2) / n;
        m.delta3 = static_cast<double>(d3) / n;
        for (std::size_t k = 0; k < bad_thresholds.size(); ++k) {
            m.bad_fraction[k] = static_cast<double>(bad_counts[k]) / n;
        }
    }
    if (gt_valid > 0) {
        m.density = static_cast<double>(evaluated) / static_cast<double>(gt_valid);
    }
    return m;
}

}  // namespace s3m
