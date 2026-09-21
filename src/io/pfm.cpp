#include "s3m/io/pfm.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

bool hostIsLittleEndian() {
    const std::uint32_t one = 1;
    std::uint8_t bytes[4];
    std::memcpy(bytes, &one, sizeof(bytes));
    return bytes[0] == 1;
}

float byteSwap(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = ((bits & 0x000000FFu) << 24) | ((bits & 0x0000FF00u) << 8) |
           ((bits & 0x00FF0000u) >> 8) | ((bits & 0xFF000000u) >> 24);
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Read one whitespace-delimited header token. Leaves the stream positioned
/// just past the single whitespace character that terminates the token, which
/// for the scale line is exactly the separator before the binary payload.
std::string readToken(std::istream& is) {
    std::string token;
    char c = 0;
    while (is.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty())
                break;
        } else {
            token.push_back(c);
        }
    }
    return token;
}

}  // namespace

cv::Mat readPfm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("readPfm: cannot open '" + path + "'");

    const std::string magic = readToken(file);
    int channels = 0;
    if (magic == "PF") {
        channels = 3;
    } else if (magic == "Pf") {
        channels = 1;
    } else {
        throw std::runtime_error("readPfm: '" + path + "' is not a PFM file (magic='" + magic +
                                 "')");
    }

    int width = 0;
    int height = 0;
    double scale = 0.0;
    try {
        width = std::stoi(readToken(file));
        height = std::stoi(readToken(file));
        scale = std::stod(readToken(file));
    } catch (const std::exception&) {
        throw std::runtime_error("readPfm: malformed header in '" + path + "'");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("readPfm: non-positive dimensions in '" + path + "'");
    }

    const bool need_swap = (scale < 0.0) != hostIsLittleEndian();
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channels;

    std::vector<float> buffer(count);
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("readPfm: truncated pixel data in '" + path + "'");
    }

    cv::Mat image(height, width, CV_MAKETYPE(CV_32F, channels));
    const std::size_t row_floats = static_cast<std::size_t>(width) * channels;
    for (int row = 0; row < height; ++row) {
        const float* src = buffer.data() + static_cast<std::size_t>(height - 1 - row) * row_floats;
        float* dst = image.ptr<float>(row);
        for (std::size_t i = 0; i < row_floats; ++i) {
            dst[i] = need_swap ? byteSwap(src[i]) : src[i];
        }
    }

    if (channels == 3)
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    return image;
}

void writePfm(const std::string& path, const cv::Mat& image) {
    CV_Assert(image.type() == CV_32FC1);
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("writePfm: cannot open '" + path + "' for writing");

    file << "Pf\n"
         << image.cols << " " << image.rows << "\n"
         << (hostIsLittleEndian() ? "-1.0" : "1.0") << "\n";

    for (int row = image.rows - 1; row >= 0; --row) {
        file.write(
            reinterpret_cast<const char*>(image.ptr<float>(row)),
            static_cast<std::streamsize>(static_cast<std::size_t>(image.cols) * sizeof(float)));
    }
}

}  // namespace s3m
