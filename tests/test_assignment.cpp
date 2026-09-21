#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "s3m/tracking/assignment.hpp"

using namespace s3m;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double totalCost(const std::vector<std::vector<double>>& cost, const std::vector<int>& row_to_col) {
    double total = 0.0;
    for (std::size_t i = 0; i < row_to_col.size(); ++i) {
        if (row_to_col[i] >= 0)
            total += cost[i][static_cast<std::size_t>(row_to_col[i])];
    }
    return total;
}

int matchCount(const std::vector<int>& row_to_col) {
    int n = 0;
    for (const int c : row_to_col) {
        if (c >= 0)
            ++n;
    }
    return n;
}

}  // namespace

TEST(Assignment, TrivialOneByOne) {
    const std::vector<std::vector<double>> cost{{3.0}};
    const Assignment a = solveAssignmentHungarian(cost, 10.0);
    ASSERT_EQ(a.row_to_col[0], 0);
    ASSERT_EQ(a.col_to_row[0], 0);
}

TEST(Assignment, OptimalBeatsTheGreedyLocalChoice) {
    // Greedily grabbing the cheapest cell (row0->col0, cost 1) forces the
    // leftover pair to cost 100; the optimal swap (row0->col1, row1->col0)
    // costs 2+2=4.
    const std::vector<std::vector<double>> cost{{1.0, 2.0}, {2.0, 100.0}};
    const Assignment a = solveAssignmentHungarian(cost, 1000.0);
    EXPECT_EQ(matchCount(a.row_to_col), 2);
    EXPECT_EQ(a.row_to_col[0], 1);
    EXPECT_EQ(a.row_to_col[1], 0);
    EXPECT_NEAR(totalCost(cost, a.row_to_col), 4.0, 1e-9);
}

TEST(Assignment, GateExcludesExpensivePairs) {
    const std::vector<std::vector<double>> cost{{1.0, 50.0}, {50.0, 1.0}};
    const Assignment a = solveAssignmentHungarian(cost, 5.0);
    EXPECT_EQ(a.row_to_col[0], 0);
    EXPECT_EQ(a.row_to_col[1], 1);
}

TEST(Assignment, OverGateFiniteEntriesAreAsForbiddenAsInfinity) {
    // Regression: an over-gate-but-finite entry must be exactly as forbidden as
    // +inf, or the "minimise total cost" objective can prefer a matching built
    // entirely from cheap-but-still-forbidden edges over one using a valid
    // gated edge (see the gate-before-padding comment in assignment.cpp).
    const std::vector<std::vector<double>> cost{
        {7.22, kInf, kInf}, {5.0, 6.46, 9.47}, {2.03, kInf, 6.56}};
    const Assignment a = solveAssignmentHungarian(cost, 4.73);
    EXPECT_EQ(matchCount(a.row_to_col), 1);
    EXPECT_EQ(a.row_to_col[2], 0);
    EXPECT_EQ(a.row_to_col[0], -1);
    EXPECT_EQ(a.row_to_col[1], -1);
}

TEST(Assignment, MaximisesMatchCountOverMinimisingCost) {
    // Matching nothing trivially costs 0; the solver must still prefer the one
    // feasible pair (cost 2.0) to leaving everything unmatched.
    const std::vector<std::vector<double>> cost{{2.0, kInf}, {kInf, kInf}};
    const Assignment a = solveAssignmentHungarian(cost, 3.0);
    EXPECT_EQ(a.row_to_col[0], 0);
}

TEST(Assignment, RectangularMoreRowsThanCols) {
    const std::vector<std::vector<double>> cost{{1.0}, {5.0}, {3.0}};
    const Assignment a = solveAssignmentHungarian(cost, 10.0);
    EXPECT_EQ(matchCount(a.row_to_col), 1);
    EXPECT_EQ(a.row_to_col[0], 0);  // the one column goes to the cheapest row
    EXPECT_EQ(a.row_to_col[1], -1);
    EXPECT_EQ(a.row_to_col[2], -1);
}

TEST(Assignment, RectangularMoreColsThanRows) {
    const std::vector<std::vector<double>> cost{{4.0, 1.0, 9.0}};
    const Assignment a = solveAssignmentHungarian(cost, 10.0);
    EXPECT_EQ(a.row_to_col[0], 1);
    EXPECT_EQ(a.col_to_row[1], 0);
    EXPECT_EQ(a.col_to_row[0], -1);
    EXPECT_EQ(a.col_to_row[2], -1);
}

TEST(Assignment, EmptyInputsAreHandled) {
    EXPECT_TRUE(solveAssignmentHungarian({}, 1.0).row_to_col.empty());
    const std::vector<std::vector<double>> zero_cols{{}};
    const Assignment a = solveAssignmentHungarian(zero_cols, 1.0);
    ASSERT_EQ(a.row_to_col.size(), 1u);
    EXPECT_EQ(a.row_to_col[0], -1);
}

TEST(Assignment, RaggedMatrixThrows) {
    const std::vector<std::vector<double>> ragged{{1.0, 2.0}, {3.0}};
    EXPECT_THROW(solveAssignmentHungarian(ragged, 10.0), std::invalid_argument);
    EXPECT_THROW(solveAssignmentGreedy(ragged, 10.0), std::invalid_argument);
}

TEST(Assignment, GreedyIsValidAndNeverBeatsHungarianOnCardinality) {
    const std::vector<std::vector<double>> cost{
        {1.0, 2.0, 9.0}, {2.0, 100.0, 3.0}, {9.0, 3.0, 1.0}};
    const double gate = 50.0;
    const Assignment h = solveAssignmentHungarian(cost, gate);
    const Assignment g = solveAssignmentGreedy(cost, gate);

    for (std::size_t i = 0; i < g.row_to_col.size(); ++i) {
        const int j = g.row_to_col[i];
        if (j < 0)
            continue;
        EXPECT_EQ(g.col_to_row[static_cast<std::size_t>(j)], static_cast<int>(i));
        EXPECT_LE(cost[i][static_cast<std::size_t>(j)], gate);
    }
    EXPECT_GE(matchCount(h.row_to_col), matchCount(g.row_to_col));
    if (matchCount(h.row_to_col) == matchCount(g.row_to_col)) {
        EXPECT_LE(totalCost(cost, h.row_to_col), totalCost(cost, g.row_to_col) + 1e-9);
    }
}

TEST(Assignment, GreedyPicksTheObviouslyCheapestPairFirst) {
    const std::vector<std::vector<double>> cost{{0.1, 5.0}, {5.0, 0.2}};
    const Assignment g = solveAssignmentGreedy(cost, 10.0);
    EXPECT_EQ(g.row_to_col[0], 0);
    EXPECT_EQ(g.row_to_col[1], 1);
}
