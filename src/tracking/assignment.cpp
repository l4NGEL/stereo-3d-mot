#include "s3m/tracking/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace s3m {
namespace {

void validateRectangular(const std::vector<std::vector<double>>& cost, int rows, int cols) {
    for (const auto& row : cost) {
        if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument("assignment: cost matrix is ragged (not rectangular)");
        }
    }
    (void)rows;
}

/// Minimum-cost PERFECT matching on a square n x n matrix (every row matched to
/// a distinct column) via the classic O(n^3) Kuhn-Munkres method with dual
/// potentials u/v. 1-indexed internally, as in the standard reference
/// implementation. Returns, for each row (0-indexed), its matched column
/// (0-indexed); always a full bijection for a square input.
std::vector<int> hungarianSquarePerfectMatching(const std::vector<std::vector<double>>& a, int n) {
    constexpr double kBig = std::numeric_limits<double>::max() / 4.0;
    std::vector<double> u(static_cast<std::size_t>(n) + 1, 0.0);
    std::vector<double> v(static_cast<std::size_t>(n) + 1, 0.0);
    std::vector<int> p(static_cast<std::size_t>(n) + 1, 0);  // p[j] = row matched to column j
    std::vector<int> way(static_cast<std::size_t>(n) + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(static_cast<std::size_t>(n) + 1, kBig);
        std::vector<char> used(static_cast<std::size_t>(n) + 1, 0);

        do {
            used[static_cast<std::size_t>(j0)] = 1;
            const int i0 = p[static_cast<std::size_t>(j0)];
            int j1 = -1;
            double delta = kBig;
            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)])
                    continue;
                const double cur =
                    a[static_cast<std::size_t>(i0 - 1)][static_cast<std::size_t>(j - 1)] -
                    u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
                if (cur < minv[static_cast<std::size_t>(j)]) {
                    minv[static_cast<std::size_t>(j)] = cur;
                    way[static_cast<std::size_t>(j)] = j0;
                }
                if (minv[static_cast<std::size_t>(j)] < delta) {
                    delta = minv[static_cast<std::size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) {
                    u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
                    v[static_cast<std::size_t>(j)] -= delta;
                } else {
                    minv[static_cast<std::size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<std::size_t>(j0)] != 0);

        do {
            const int j1 = way[static_cast<std::size_t>(j0)];
            p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> col_of_row(static_cast<std::size_t>(n), -1);
    for (int j = 1; j <= n; ++j) {
        col_of_row[static_cast<std::size_t>(p[static_cast<std::size_t>(j)] - 1)] = j - 1;
    }
    return col_of_row;
}

}  // namespace

Assignment solveAssignmentHungarian(const std::vector<std::vector<double>>& cost, double gate) {
    const int rows = static_cast<int>(cost.size());
    const int cols = rows > 0 ? static_cast<int>(cost.front().size()) : 0;
    validateRectangular(cost, rows, cols);

    Assignment result;
    result.row_to_col.assign(static_cast<std::size_t>(rows), -1);
    result.col_to_row.assign(static_cast<std::size_t>(cols), -1);
    if (rows == 0 || cols == 0)
        return result;

    // A value strictly worse than any real (even ungated) cost, so the solver
    // only ever uses a padding/forbidden cell when no better option remains.
    double big = gate;
    for (const auto& row : cost) {
        for (const double c : row) {
            if (std::isfinite(c))
                big = std::max(big, c);
        }
    }
    big = big * 4.0 + 1.0;

    // Square, and -- critically -- gate BEFORE padding: an over-gate but finite
    // entry must be exactly as forbidden as +inf here. Otherwise the "minimise
    // total cost" objective can prefer a matching built entirely from cheap
    // ungated-but-still-forbidden edges over one that uses a valid gated edge,
    // since it has no other way to know that edge is off-limits.
    const int n = std::max(rows, cols);
    std::vector<std::vector<double>> square(static_cast<std::size_t>(n),
                                            std::vector<double>(static_cast<std::size_t>(n), big));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const double c = cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (std::isfinite(c) && c <= gate) {
                square[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = c;
            }
        }
    }

    const std::vector<int> col_of_row = hungarianSquarePerfectMatching(square, n);
    for (int i = 0; i < rows; ++i) {
        const int j = col_of_row[static_cast<std::size_t>(i)];
        if (j >= 0 && j < cols) {
            const double c = cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (std::isfinite(c) && c <= gate) {
                result.row_to_col[static_cast<std::size_t>(i)] = j;
                result.col_to_row[static_cast<std::size_t>(j)] = i;
            }
        }
    }
    return result;
}

Assignment solveAssignmentGreedy(const std::vector<std::vector<double>>& cost, double gate) {
    const int rows = static_cast<int>(cost.size());
    const int cols = rows > 0 ? static_cast<int>(cost.front().size()) : 0;
    validateRectangular(cost, rows, cols);

    Assignment result;
    result.row_to_col.assign(static_cast<std::size_t>(rows), -1);
    result.col_to_row.assign(static_cast<std::size_t>(cols), -1);

    struct Edge {
        double cost;
        int row;
        int col;
    };
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const double c = cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            if (std::isfinite(c) && c <= gate)
                edges.push_back({c, i, j});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.cost < b.cost; });

    std::vector<char> row_used(static_cast<std::size_t>(rows), 0);
    std::vector<char> col_used(static_cast<std::size_t>(cols), 0);
    for (const Edge& e : edges) {
        if (row_used[static_cast<std::size_t>(e.row)] || col_used[static_cast<std::size_t>(e.col)])
            continue;
        row_used[static_cast<std::size_t>(e.row)] = 1;
        col_used[static_cast<std::size_t>(e.col)] = 1;
        result.row_to_col[static_cast<std::size_t>(e.row)] = e.col;
        result.col_to_row[static_cast<std::size_t>(e.col)] = e.row;
    }
    return result;
}

}  // namespace s3m
