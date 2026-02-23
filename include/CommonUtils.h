#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace CommonUtils {

inline std::string trim(std::string_view s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r\n");
    return std::string(s.substr(b, e - b + 1));
}

inline std::string toLower(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

inline double medianByNth(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + (mid - 1), values.begin() + mid);
        return (values[mid - 1] + upper) / 2.0;
    }
    return upper;
}

inline double quantileByNth(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    if (q <= 0.0) return *std::min_element(values.begin(), values.end());
    if (q >= 1.0) return *std::max_element(values.begin(), values.end());

    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));

    std::nth_element(values.begin(), values.begin() + lo, values.end());
    const double loVal = values[lo];
    if (hi == lo) return loVal;

    std::nth_element(values.begin(), values.begin() + hi, values.end());
    const double hiVal = values[hi];
    const double frac = pos - static_cast<double>(lo);
    return loVal * (1.0 - frac) + hiVal * frac;
}

} // namespace CommonUtils
