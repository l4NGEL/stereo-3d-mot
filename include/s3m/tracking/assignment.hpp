#pragma once

#include <vector>

namespace s3m {

/// Result of solving a (possibly rectangular) assignment problem.
struct Assignment {
    std::vector<int> row_to_col;  ///< size = rows; -1 if that row is unmatched
    std::vector<int> col_to_row;  ///< size = cols; -1 if that column is unmatched
};

/// Minimum-cost bipartite matching (Hungarian / Kuhn-Munkres, O(n^3) where
/// n = max(rows, cols)).
///
/// `cost[i][j] > gate` (this includes +inf) forbids pairing row i with column
/// j. Among all matchings that respect the gate, the solver first maximises the
/// number of matched pairs, then minimises their total cost -- i.e. it never
/// leaves a feasible pair unmatched merely to save cost elsewhere. `cost` must
/// be rectangular (every row the same length); throws std::invalid_argument
/// otherwise.
Assignment solveAssignmentHungarian(const std::vector<std::vector<double>>& cost, double gate);

/// Greedy nearest-neighbour baseline: repeatedly take the globally cheapest
/// still-available (row, col) pair with cost <= gate. O(rows*cols*log(rows*cols)),
/// not guaranteed optimal -- exists to contrast against the Hungarian solver
/// (see benchmark_track).
Assignment solveAssignmentGreedy(const std::vector<std::vector<double>>& cost, double gate);

}  // namespace s3m
