#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/core/config.hpp"

using namespace s3m;

TEST(Config, Defaults) {
    const Config cfg = Config::defaults();
    EXPECT_EQ(cfg.stereo_matcher.type, "SGBM");
    EXPECT_EQ(cfg.stereo_matcher.num_disparities, 128);
    EXPECT_EQ(cfg.stereo_matcher.block_size, 5);
    EXPECT_EQ(cfg.tracking.max_age, 30);
    EXPECT_EQ(cfg.tracking.min_hits, 3);
}

TEST(Config, LoadsOverridesAndKeepsOtherDefaults) {
    const std::string path = "s3m_test_cfg.yaml";
    {
        // Author the file with FileStorage so the on-disk syntax is exactly
        // what the parser must accept; this exercises Config's field mapping.
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "stereo_matcher" << "{" << "type" << "BM" << "num_disparities" << 64 << "mode_hh" << 1
           << "}";
        fs << "tracking" << "{" << "max_age" << 12 << "}";
        fs << "depth_eval" << "{" << "bad_thresholds" << "[" << 0.5 << 1.0 << 3.0 << "]" << "}";
    }

    const Config cfg = Config::load(path);
    std::remove(path.c_str());

    EXPECT_EQ(cfg.stereo_matcher.type, "BM");
    EXPECT_EQ(cfg.stereo_matcher.num_disparities, 64);
    EXPECT_TRUE(cfg.stereo_matcher.mode_hh);
    EXPECT_EQ(cfg.tracking.max_age, 12);

    ASSERT_EQ(cfg.depth_eval.bad_thresholds.size(), 3u);
    EXPECT_DOUBLE_EQ(cfg.depth_eval.bad_thresholds[0], 0.5);
    EXPECT_DOUBLE_EQ(cfg.depth_eval.bad_thresholds[2], 3.0);

    // untouched keys keep their defaults
    EXPECT_EQ(cfg.stereo_matcher.block_size, 5);
    EXPECT_EQ(cfg.tracking.min_hits, 3);
}

TEST(Config, ParsesShippedDefaultYaml) {
    // The repo's configs/default.yaml is hand-written; make sure it still parses
    // and matches the documented defaults. Skips cleanly if run from elsewhere.
    std::ifstream probe("configs/default.yaml");
    if (!probe.good())
        GTEST_SKIP() << "configs/default.yaml not reachable from CWD";

    const Config cfg = Config::load("configs/default.yaml");
    EXPECT_EQ(cfg.stereo_matcher.type, "SGBM");
    EXPECT_EQ(cfg.stereo_matcher.num_disparities, 128);
    EXPECT_EQ(cfg.tracking.max_age, 30);
    EXPECT_FALSE(cfg.depth_eval.bad_thresholds.empty());
}

TEST(Config, ThrowsOnMissingFile) {
    EXPECT_THROW(Config::load("s3m_missing_config_54321.yaml"), std::runtime_error);
}
