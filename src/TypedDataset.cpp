#include "TypedDataset.h"
#include "CSVUtils.h"
#include "CommonUtils.h"
#include "SeldonExceptions.h"
#include <algorithm>
#include <cstdio>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

TypedDataset::TypedDataset(std::string filename, char delimiter)
    : filename_(std::move(filename)), delimiter_(delimiter) {}

namespace {
bool isMissingToken(const std::string& raw) {
    std::string s = CommonUtils::trim(raw);
    if (s.empty()) return true;
    s = CommonUtils::toLower(std::move(s));
    return s == "na" || s == "n/a" || s == "null" || s == "none" || s == "nan" || s == "missing";
}

bool parseFixedInt(const std::string& s, size_t offset, size_t len, int& out) {
    if (offset + len > s.size()) return false;
    int value = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char ch = static_cast<unsigned char>(s[offset + i]);
        if (ch < '0' || ch > '9') return false;
        value = value * 10 + static_cast<int>(ch - '0');
    }
    out = value;
    return true;
}

bool isLeapYear(int year) {
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    return (year % 4 == 0);
}

int daysInMonth(int year, int month) {
    static const int kMonthDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) return 0;
    if (month == 2) return isLeapYear(year) ? 29 : 28;
    return kMonthDays[month - 1];
}

int64_t daysFromCivil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const int mp = static_cast<int>(m) + (m > 2 ? -3 : 9);
    const unsigned doy = (153 * static_cast<unsigned>(mp) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

bool parseTimePart(const std::string& timePart, int& hour, int& minute, int& second) {
    if (timePart.empty()) {
        hour = minute = second = 0;
        return true;
    }
    if (timePart.size() != 8) return false;
    if (!parseFixedInt(timePart, 0, 2, hour) || timePart[2] != ':' ||
        !parseFixedInt(timePart, 3, 2, minute) || timePart[5] != ':' ||
        !parseFixedInt(timePart, 6, 2, second)) {
        return false;
    }
    return true;
}

bool parseDatePart(const std::string& datePart, TypedDataset::DateLocaleHint localeHint, int& year, int& month, int& day) {
    // ISO: YYYY-MM-DD
    if (datePart.size() == 10 && datePart[4] == '-' && datePart[7] == '-') {
        return parseFixedInt(datePart, 0, 4, year) &&
               parseFixedInt(datePart, 5, 2, month) &&
               parseFixedInt(datePart, 8, 2, day);
    }

    // Slash format: dd/mm/yyyy or mm/dd/yyyy based on locale hint.
    if (datePart.size() == 10 && datePart[2] == '/' && datePart[5] == '/') {
        int a = 0;
        int b = 0;
        int y = 0;
        if (!parseFixedInt(datePart, 0, 2, a) ||
            !parseFixedInt(datePart, 3, 2, b) ||
            !parseFixedInt(datePart, 6, 4, y)) {
            return false;
        }

        if (localeHint == TypedDataset::DateLocaleHint::DMY) {
            day = a;
            month = b;
        } else if (localeHint == TypedDataset::DateLocaleHint::MDY) {
            month = a;
            day = b;
        } else {
            if (a > 12 && b <= 12) {
                day = a;
                month = b;
            } else {
                month = a;
                day = b;
            }
        }
        year = y;
        return true;
    }

    // DMY: DD-MM-YYYY
    if (datePart.size() == 10 && datePart[2] == '-' && datePart[5] == '-') {
        return parseFixedInt(datePart, 0, 2, day) &&
               parseFixedInt(datePart, 3, 2, month) &&
               parseFixedInt(datePart, 6, 4, year);
    }

    // MDY two-digit year: MM-DD-YY
    if (datePart.size() == 8 && datePart[2] == '-' && datePart[5] == '-') {
        int yy = 0;
        if (!parseFixedInt(datePart, 0, 2, month) ||
            !parseFixedInt(datePart, 3, 2, day) ||
            !parseFixedInt(datePart, 6, 2, yy)) {
            return false;
        }
        year = (yy >= 70) ? (1900 + yy) : (2000 + yy);
        return true;
    }

    // ISO week date: YYYY-Www-D (D:1..7)
    if (datePart.size() == 10 && datePart[4] == '-' && datePart[5] == 'W' && datePart[8] == '-') {
        int isoYear = 0;
        int isoWeek = 0;
        int isoDay = 0;
        if (!parseFixedInt(datePart, 0, 4, isoYear) ||
            !parseFixedInt(datePart, 6, 2, isoWeek) ||
            !parseFixedInt(datePart, 9, 1, isoDay)) {
            return false;
        }
        if (isoWeek < 1 || isoWeek > 53 || isoDay < 1 || isoDay > 7) return false;

        const int64_t jan4 = daysFromCivil(isoYear, 1, 4);
        const int jan4WeekdayMon1 = static_cast<int>(((jan4 + 3) % 7 + 7) % 7) + 1;
        const int64_t isoWeek1Monday = jan4 - static_cast<int64_t>(jan4WeekdayMon1 - 1);
        const int64_t targetDays = isoWeek1Monday + static_cast<int64_t>((isoWeek - 1) * 7 + (isoDay - 1));

        int z = static_cast<int>(targetDays + 719468);
        const int era = (z >= 0 ? z : z - 146096) / 146097;
        const unsigned doe = static_cast<unsigned>(z - era * 146097);
        const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        int y = static_cast<int>(yoe) + era * 400;
        const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const unsigned mp = (5 * doy + 2) / 153;
        const unsigned d = doy - (153 * mp + 2) / 5 + 1;
        const unsigned m = mp + (mp < 10 ? 3 : -9);
        y += (m <= 2);
        year = y;
        month = static_cast<int>(m);
        day = static_cast<int>(d);
        return true;
    }

    return false;
}

std::string findExecutableInPath(const std::string& command) {
    if (command.empty()) return "";
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return "";

    std::string pathValue(pathEnv);
    std::stringstream ss(pathValue);
    std::string token;
    while (std::getline(ss, token, ':')) {
        if (token.empty()) token = ".";
        std::filesystem::path candidate = std::filesystem::path(token) / command;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec && ::access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return "";
}

bool commandAvailable(const std::string& command) {
    return !findExecutableInPath(command).empty();
}

int spawnToFile(const std::string& executable,
                const std::vector<std::string>& args,
                const std::string& outputPath) {
    const int outFd = ::open(outputPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (outFd < 0) return -1;

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(outFd);
        return -1;
    }

    if (pid == 0) {
        const int devNull = ::open("/dev/null", O_WRONLY);
        if (devNull >= 0) {
            ::dup2(devNull, STDERR_FILENO);
            ::close(devNull);
        }

        if (::dup2(outFd, STDOUT_FILENO) < 0) {
            _exit(127);
        }
        ::close(outFd);

        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);

        ::execv(executable.c_str(), argv.data());
        _exit(127);
    }

    ::close(outFd);
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::pair<std::string, bool> resolveDatasetInputPath(const std::string& sourcePath) {
    namespace fs = std::filesystem;
    const fs::path src(sourcePath);
    std::string lowerExt = CommonUtils::toLower(src.extension().string());

    if (lowerExt == ".csv") {
        return {sourcePath, false};
    }

    const fs::path tmpBase = fs::temp_directory_path() / ("seldon_input_" + std::to_string(static_cast<unsigned long long>(std::hash<std::string>{}(sourcePath + std::to_string(std::time(nullptr))))) + ".csv");
    const std::string tmpPath = tmpBase.string();

    auto runToTmp = [&](const std::string& executable, const std::vector<std::string>& args) {
        const int rc = spawnToFile(executable, args, tmpPath);
        if (rc != 0) {
            std::error_code ec;
            std::filesystem::remove(tmpPath, ec);
            throw Seldon::DatasetException("Failed to convert input source: " + sourcePath);
        }
        return std::make_pair(tmpPath, true);
    };

    if (lowerExt == ".gz") {
        const std::string exe = findExecutableInPath("gzip");
        if (exe.empty()) throw Seldon::DatasetException("gzip is required to read .gz files");
        return runToTmp(exe, {"-cd", sourcePath});
    }
    if (lowerExt == ".zip") {
        const std::string exe = findExecutableInPath("unzip");
        if (exe.empty()) throw Seldon::DatasetException("unzip is required to read .zip files");
        return runToTmp(exe, {"-p", sourcePath});
    }
    if (lowerExt == ".xlsx") {
        const std::string exe = findExecutableInPath("xlsx2csv");
        if (!exe.empty()) {
            return runToTmp(exe, {sourcePath});
        }
        throw Seldon::DatasetException("XLSX import requires xlsx2csv. See docs/ENABLE_EXCEL_IMPORT.md");
    }
    if (lowerExt == ".xls") {
        const std::string exe = findExecutableInPath("xls2csv");
        if (!exe.empty()) {
            return runToTmp(exe, {sourcePath});
        }
        throw Seldon::DatasetException("XLS import requires xls2csv (libxls tools). See docs/ENABLE_EXCEL_IMPORT.md");
    }

    return {sourcePath, false};
}

std::string normalizeNumericToken(const std::string& input, TypedDataset::NumericSeparatorPolicy policy) {
    std::string cleaned;
    cleaned.reserve(input.size());
    for (char ch : input) {
        if (!std::isspace(static_cast<unsigned char>(ch)) && ch != '_') {
            cleaned.push_back(ch);
        }
    }

    const auto toUS = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char ch : s) {
            if (ch != ',') out.push_back(ch);
        }
        return out;
    };

    const auto toEuropean = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char ch : s) {
            if (ch == '.') {
                continue;
            }
            if (ch == ',') {
                out.push_back('.');
            } else {
                out.push_back(ch);
            }
        }
        return out;
    };

    if (policy == TypedDataset::NumericSeparatorPolicy::US_THOUSANDS) {
        return toUS(cleaned);
    }
    if (policy == TypedDataset::NumericSeparatorPolicy::EUROPEAN) {
        return toEuropean(cleaned);
    }

    const size_t dotPos = cleaned.find('.');
    const size_t commaPos = cleaned.find(',');
    if (dotPos != std::string::npos && commaPos != std::string::npos) {
        const size_t lastDot = cleaned.find_last_of('.');
        const size_t lastComma = cleaned.find_last_of(',');
        if (lastComma > lastDot) {
            return toEuropean(cleaned);
        }
        return toUS(cleaned);
    }

    if (commaPos != std::string::npos) {
        const size_t commaCount = static_cast<size_t>(std::count(cleaned.begin(), cleaned.end(), ','));
        const bool singleComma = (commaCount == 1);
        const size_t digitsAfter = cleaned.size() - commaPos - 1;
        if (singleComma && digitsAfter >= 1 && digitsAfter <= 4) {
            const bool likelyThousands = (digitsAfter == 3 && commaPos > 0 && std::isdigit(static_cast<unsigned char>(cleaned[commaPos - 1])));
            if (!likelyThousands) {
                std::string out = cleaned;
                out[commaPos] = '.';
                return out;
            }
        }
        return toUS(cleaned);
    }

    if (dotPos != std::string::npos) {
        const size_t dotCount = static_cast<size_t>(std::count(cleaned.begin(), cleaned.end(), '.'));
        if (dotCount > 1) {
            return toUS(cleaned);
        }
    }

    return cleaned;
}

bool parseIntegerEpochSeconds(const std::string& input, int64_t& out) {
    if (input.empty()) return false;
    const char* begin = input.data();
    const char* end = begin + input.size();
    auto [ptr, ec] = std::from_chars(begin, end, out, 10);
    if (ec != std::errc{} || ptr != end) return false;
    constexpr int64_t kEpochMin = 0;           // 1970-01-01T00:00:00Z
    constexpr int64_t kEpochMax = 2524608000;  // 2050-01-01T00:00:00Z
    return out >= kEpochMin && out <= kEpochMax;
}

bool parseLikelyEpochSeconds(const std::string& input, int64_t& out) {
    const std::string s = CommonUtils::trim(input);
    if (s.size() < 9 || s.size() > 10) return false;
    if (!std::all_of(s.begin(), s.end(), [](unsigned char ch) { return std::isdigit(ch); })) return false;
    return parseIntegerEpochSeconds(s, out);
}

bool looksNumericLike(const std::string& token) {
    if (token.empty()) return false;

    size_t total = 0;
    size_t numericLike = 0;
    size_t digits = 0;
    for (char ch : token) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isspace(uch)) continue;
        ++total;
        if (std::isdigit(uch)) {
            ++digits;
            ++numericLike;
            continue;
        }
        switch (ch) {
            case '.':
            case ',':
            case '+':
            case '-':
            case '%':
            case '$':
            case '(':
            case ')':
            case '_':
                ++numericLike;
                break;
            default:
                break;
        }
    }

    if (digits == 0 || total == 0) return false;
    return (static_cast<double>(numericLike) / static_cast<double>(total)) >= 0.80;
}

std::string hardCoerceNumericToken(const std::string& input) {
    std::string s = CommonUtils::trim(input);
    if (s.empty()) return s;

    bool negative = false;
    if (s.size() >= 2 && s.front() == '(' && s.back() == ')') {
        negative = true;
        s = CommonUtils::trim(s.substr(1, s.size() - 2));
    }

    std::string filtered;
    filtered.reserve(s.size() + 1);
    bool seenSign = false;
    bool seenExponent = false;
    bool seenExponentSign = false;

    for (char ch : s) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isspace(uch)) continue;

        if ((ch == '+' || ch == '-') && !seenSign) {
            filtered.push_back(ch);
            seenSign = true;
            continue;
        }

        if ((ch == 'e' || ch == 'E') && !seenExponent) {
            if (!filtered.empty()) {
                filtered.push_back(ch);
                seenExponent = true;
                seenExponentSign = false;
                continue;
            }
        }

        if ((ch == '+' || ch == '-') && seenExponent && !seenExponentSign && !filtered.empty()) {
            const char prev = filtered.back();
            if (prev == 'e' || prev == 'E') {
                filtered.push_back(ch);
                seenExponentSign = true;
                continue;
            }
        }

        if (std::isdigit(uch) || ch == '.' || ch == ',' || ch == '_') {
            filtered.push_back(ch);
        }
    }

    if (filtered.empty()) return filtered;
    if (negative && filtered.front() != '-') {
        filtered.insert(filtered.begin(), '-');
    }
    return filtered;
}

bool isSkippableControlRow(const std::vector<std::string>& row) {
    if (row.empty()) return true;
    size_t nonEmpty = 0;
    for (const auto& cell : row) {
        if (!CommonUtils::trim(cell).empty()) {
            ++nonEmpty;
        }
    }
    if (nonEmpty == 0) return true;
    if (nonEmpty > 1) return false;

    const std::string marker = CommonUtils::toLower(CommonUtils::trim(row.front()));
    if (marker.empty()) return true;
    return marker.rfind("--", 0) == 0 ||
           marker.find("data gap") != std::string::npos ||
           marker.find("maintenance log") != std::string::npos;
}

bool containsMetadataKeyword(const std::string& token) {
    const std::string t = CommonUtils::toLower(CommonUtils::trim(token));
    if (t.empty()) return false;
    return t.find("total") != std::string::npos ||
           t.find("subtotal") != std::string::npos ||
           t.find("grand total") != std::string::npos ||
           t.find("summary") != std::string::npos ||
           t.find("rollup") != std::string::npos ||
           t.find("aggregate") != std::string::npos ||
           t.find("stop:") != std::string::npos;
}
}

bool TypedDataset::parseDouble(const std::string& v, double& out) const {
    auto sv = CommonUtils::trim(v);
    if (isMissingToken(sv)) return false;

    std::string cleaned = normalizeNumericToken(sv, numericSeparatorPolicy_);

    if (!cleaned.empty() && cleaned.front() == '+') {
        cleaned.erase(cleaned.begin());
    }
    if (!cleaned.empty() && cleaned.back() == '%') {
        cleaned.pop_back();
    }

    if (cleaned.empty()) return false;

    const char* b = cleaned.data();
    const char* e = b + cleaned.size();
    auto [p, ec] = std::from_chars(b, e, out, std::chars_format::general);
    if (ec == std::errc{} && p == e && std::isfinite(out)) return true;

    if (!looksNumericLike(sv)) return false;

    std::string hardCoerced = hardCoerceNumericToken(sv);
    if (hardCoerced.empty()) return false;
    cleaned = normalizeNumericToken(hardCoerced, numericSeparatorPolicy_);
    if (!cleaned.empty() && cleaned.front() == '+') {
        cleaned.erase(cleaned.begin());
    }
    if (!cleaned.empty() && cleaned.back() == '%') {
        cleaned.pop_back();
    }
    if (cleaned.empty()) return false;

    b = cleaned.data();
    e = b + cleaned.size();
    auto [p2, ec2] = std::from_chars(b, e, out, std::chars_format::general);
    return ec2 == std::errc{} && p2 == e && std::isfinite(out);
}

bool TypedDataset::parseDateTime(const std::string& v, int64_t& outUnixSeconds) const {
    std::string s = CommonUtils::trim(v);
    if (s.empty() || isMissingToken(s)) return false;

    if (parseIntegerEpochSeconds(s, outUnixSeconds)) {
        return true;
    }

    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;

    std::string datePart = s;
    std::string timePart;
    const size_t sep = s.find(' ');
    if (sep != std::string::npos) {
        datePart = s.substr(0, sep);
        timePart = s.substr(sep + 1);
    }

    if (!parseDatePart(datePart, dateLocaleHint_, year, month, day)) {
        return false;
    }
    if (!parseTimePart(timePart, hour, minute, second)) {
        return false;
    }

    if (month < 1 || month > 12) return false;
    const int dim = daysInMonth(year, month);
    if (day < 1 || day > dim) return false;
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 60) return false;

    const int64_t days = daysFromCivil(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
    outUnixSeconds = days * 86400 + static_cast<int64_t>(hour) * 3600 + static_cast<int64_t>(minute) * 60 + second;
    return true;
}

void TypedDataset::load() {
    auto [resolvedPath, isTemporary] = resolveDatasetInputPath(filename_);
    struct TempFileGuard {
        std::string path;
        bool enabled = false;
        ~TempFileGuard() {
            if (!enabled) return;
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } tempGuard{resolvedPath, isTemporary};

    std::ifstream in(resolvedPath, std::ios::binary);
    if (!in) throw Seldon::IOException("Could not open file: " + resolvedPath);

    CSVUtils::skipBOM(in);

    bool malformed = false;
    auto header = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
    if (malformed || header.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");
    header = CSVUtils::normalizeHeader(header);

    std::vector<uint8_t> forcedTypeMask(header.size(), static_cast<uint8_t>(0));
    std::vector<ColumnType> forcedType(header.size(), ColumnType::CATEGORICAL);
    for (size_t c = 0; c < header.size(); ++c) {
        const std::string key = CommonUtils::toLower(CommonUtils::trim(header[c]));
        const auto it = columnTypeOverrides_.find(key);
        if (it == columnTypeOverrides_.end()) continue;
        forcedTypeMask[c] = static_cast<uint8_t>(1);
        forcedType[c] = it->second;
    }

    std::vector<size_t> numericHits(header.size(), 0);
    std::vector<size_t> numericLikeHits(header.size(), 0);
    std::vector<size_t> datetimeHits(header.size(), 0);
    std::vector<size_t> nonMissing(header.size(), 0);
    rowCount_ = 0;
    std::vector<uint8_t> acceptedRowMask;
    acceptedRowMask.reserve(8192);

    std::vector<size_t> dateLikeHeaderIndices;
    dateLikeHeaderIndices.reserve(header.size());
    std::vector<uint8_t> dateLikeHeaderMask(header.size(), static_cast<uint8_t>(0));
    for (size_t c = 0; c < header.size(); ++c) {
        const std::string lower = CommonUtils::toLower(header[c]);
        if (lower.find("date") != std::string::npos ||
            lower.find("time") != std::string::npos ||
            lower.find("timestamp") != std::string::npos) {
            dateLikeHeaderIndices.push_back(c);
            dateLikeHeaderMask[c] = static_cast<uint8_t>(1);
        }
    }

    auto reconcileRowWidth = [&](std::vector<std::string>& row) {
        if (row.size() <= header.size()) {
            row.resize(header.size());
            return;
        }

        auto semanticFieldScore = [&](size_t colIdx, const std::string& token) {
            if (colIdx >= header.size()) return 0;
            const std::string h = CommonUtils::toLower(header[colIdx]);
            const std::string t = CommonUtils::trim(token);
            if (t.empty()) return 0;

            double dv = 0.0;
            int64_t tv = 0;
            const bool isNum = parseDouble(t, dv);
            const bool isTime = parseDateTime(t, tv);

            int score = 0;
            const bool expectsTime = h.find("date") != std::string::npos || h.find("time") != std::string::npos || h.find("timestamp") != std::string::npos;
            const bool expectsNumeric = h.find("amount") != std::string::npos ||
                                        h.find("price") != std::string::npos ||
                                        h.find("cost") != std::string::npos ||
                                        h.find("total") != std::string::npos ||
                                        h.find("revenue") != std::string::npos ||
                                        h.find("qty") != std::string::npos ||
                                        h.find("count") != std::string::npos ||
                                        h == "id";
            const bool expectsCategory = h.find("status") != std::string::npos || h.find("state") != std::string::npos || h.find("category") != std::string::npos || h.find("notes") != std::string::npos || h.find("name") != std::string::npos;

            if (expectsTime) {
                if (isTime) score += 4;
                if (isNum) score -= 2;
            }
            if (expectsNumeric) {
                if (isNum) score += 3;
                if (isTime) score -= 1;
            }
            if (expectsCategory) {
                if (!isNum && !isTime) score += 2;
                if (isNum && h.find("id") == std::string::npos) score -= 1;
            }

            if (t.find('{') != std::string::npos || t.find('}') != std::string::npos || t.find('[') != std::string::npos || t.find(']') != std::string::npos) {
                score += expectsCategory ? 2 : 0;
            }
            return score;
        };

        while (row.size() > header.size()) {
            if (row.size() < 2) {
                break;
            }
            size_t bestIndex = std::numeric_limits<size_t>::max();
            int bestScore = -1;

            for (size_t i = 0; i + 1 < row.size(); ++i) {
                const std::string merged = row[i] + delimiter_ + row[i + 1];
                int score = 0;

                double numericValue = 0.0;
                int64_t datetimeValue = 0;
                if (parseDouble(merged, numericValue)) {
                    score += 4;
                }
                if (parseDateTime(merged, datetimeValue)) {
                    score += 3;
                }

                bool leftHasDigit = std::any_of(row[i].begin(), row[i].end(), [](unsigned char ch) { return std::isdigit(ch); });
                bool rightHasDigit = std::any_of(row[i + 1].begin(), row[i + 1].end(), [](unsigned char ch) { return std::isdigit(ch); });
                if (leftHasDigit && rightHasDigit) {
                    score += 1;
                }

                score += semanticFieldScore(i, merged);
                if (i + 2 < row.size()) {
                    score += semanticFieldScore(i + 1, row[i + 2]);
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            if (bestIndex == std::numeric_limits<size_t>::max()) {
                if (header.empty() || row.size() < 2) {
                    break;
                }
                bestIndex = std::min(header.size() - 1, row.size() - 2);
            }

            row[bestIndex] += delimiter_;
            row[bestIndex] += row[bestIndex + 1];
            row.erase(row.begin() + static_cast<std::ptrdiff_t>(bestIndex + 1));
        }
    };

    auto repairDateShiftedRow = [&](std::vector<std::string>& row) {
        if (row.size() != header.size()) return;
        if (dateLikeHeaderIndices.empty()) return;

        for (size_t dateIdx : dateLikeHeaderIndices) {
            if (dateIdx == 0 || dateIdx + 1 >= row.size()) continue;

            int64_t currentTs = 0;
            int64_t nextTs = 0;
            if (parseDateTime(row[dateIdx], currentTs)) continue;
            if (!parseDateTime(row[dateIdx + 1], nextTs)) continue;

            row[dateIdx - 1] += delimiter_;
            row[dateIdx - 1] += row[dateIdx];
            row.erase(row.begin() + static_cast<std::ptrdiff_t>(dateIdx));
            row.push_back(std::string());
            break;
        }
    };

    auto isLikelyMetadataRow = [&](const std::vector<std::string>& row, const std::vector<double>& runningSums, size_t acceptedRows) {
        if (row.empty()) return true;

        for (const auto& cell : row) {
            if (containsMetadataKeyword(cell)) return true;
        }

        if (acceptedRows < 3 || runningSums.size() != header.size()) {
            return false;
        }

        size_t nonMissingCells = 0;
        size_t numericCells = 0;
        size_t sumAlignedCells = 0;
        for (size_t c = 0; c < header.size(); ++c) {
            const std::string cell = (c < row.size()) ? row[c] : std::string();
            if (isMissingToken(cell)) continue;
            ++nonMissingCells;

            double dv = 0.0;
            if (!parseDouble(cell, dv)) continue;
            ++numericCells;

            const double baseline = std::max(1.0, std::abs(runningSums[c]));
            const double tolerance = baseline * 1e-6;
            if (std::abs(dv - runningSums[c]) <= tolerance) {
                ++sumAlignedCells;
            }
        }

        if (numericCells == 0) return false;
        if (nonMissingCells > (header.size() / 2 + 1)) return false;
        return sumAlignedCells >= 1 && (sumAlignedCells * 2 >= numericCells);
    };

    auto updateRunningSums = [&](const std::vector<std::string>& row, std::vector<double>& runningSums) {
        for (size_t c = 0; c < header.size(); ++c) {
            const std::string cell = (c < row.size()) ? row[c] : std::string();
            double dv = 0.0;
            if (parseDouble(cell, dv)) {
                runningSums[c] += dv;
            }
        }
    };

    std::vector<double> runningSums(header.size(), 0.0);

    while (in.peek() != EOF) {
        auto row = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
        if (row.empty() || malformed || isSkippableControlRow(row)) continue;
        reconcileRowWidth(row);
        repairDateShiftedRow(row);
        const bool accepted = !isLikelyMetadataRow(row, runningSums, rowCount_);
        acceptedRowMask.push_back(static_cast<uint8_t>(accepted ? 1 : 0));
        if (!accepted) continue;
        ++rowCount_;

        const size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < cols; ++c) {
            if (isMissingToken(row[c])) continue;
            ++nonMissing[c];
            double dv = 0.0;
            int64_t tv = 0;
            if (dateLikeHeaderMask[c]) {
                if (parseDateTime(row[c], tv)) {
                    ++datetimeHits[c];
                } else if (parseDouble(row[c], dv)) {
                    ++numericHits[c];
                } else if (looksNumericLike(row[c])) {
                    ++numericLikeHits[c];
                }
            } else {
                if (parseDouble(row[c], dv)) {
                    ++numericHits[c];
                } else if (looksNumericLike(row[c])) {
                    ++numericLikeHits[c];
                } else if (parseDateTime(row[c], tv)) {
                    ++datetimeHits[c];
                }
            }
        }
        updateRunningSums(row, runningSums);
    }

    columns_.clear();
    columns_.reserve(header.size());

    for (size_t c = 0; c < header.size(); ++c) {
        TypedColumn col;
        col.name = header[c];
        col.missing.assign(rowCount_, static_cast<uint8_t>(0));

        if (forcedTypeMask[c]) {
            col.type = forcedType[c];
            if (col.type == ColumnType::NUMERIC) {
                col.values = std::vector<double>(rowCount_, std::numeric_limits<double>::quiet_NaN());
            } else if (col.type == ColumnType::DATETIME) {
                col.values = std::vector<int64_t>(rowCount_, 0);
            } else {
                col.values = std::vector<std::string>(rowCount_);
            }
            columns_.push_back(std::move(col));
            continue;
        }

        const bool strongNumeric = nonMissing[c] > 0 && numericHits[c] >= std::max<size_t>(3, (nonMissing[c] * 8) / 10);
        const bool coercibleNumeric = nonMissing[c] > 0 &&
                                      numericLikeHits[c] >= std::max<size_t>(3, (nonMissing[c] * 9) / 10) &&
                                      numericHits[c] >= std::max<size_t>(2, nonMissing[c] / 3);
        const size_t datetimeThreshold = dateLikeHeaderMask[c]
            ? std::max<size_t>(3, (nonMissing[c] * 6) / 10)
            : std::max<size_t>(3, (nonMissing[c] * 8) / 10);
        const bool strongDatetime = nonMissing[c] > 0 && datetimeHits[c] >= datetimeThreshold;

        if (dateLikeHeaderMask[c] && strongDatetime) {
            col.type = ColumnType::DATETIME;
            col.values = std::vector<int64_t>(rowCount_, 0);
        } else if (strongNumeric || coercibleNumeric) {
            col.type = ColumnType::NUMERIC;
            col.values = std::vector<double>(rowCount_, std::numeric_limits<double>::quiet_NaN());
        } else {
            if (strongDatetime) {
                col.type = ColumnType::DATETIME;
                col.values = std::vector<int64_t>(rowCount_, 0);
            } else {
                col.type = ColumnType::CATEGORICAL;
                col.values = std::vector<std::string>(rowCount_);
            }
        }
        columns_.push_back(std::move(col));
    }

    std::vector<size_t> datetimeObserved(columns_.size(), 0);
    std::vector<size_t> datetimeParseFailures(columns_.size(), 0);

    in.clear();
    in.seekg(0, std::ios::beg);
    CSVUtils::skipBOM(in);
    malformed = false;
    auto headerSecondPass = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
    if (malformed || headerSecondPass.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");

    size_t r = 0;
    size_t decisionIdx = 0;
    std::fill(runningSums.begin(), runningSums.end(), 0.0);
    while (in.peek() != EOF) {
        auto row = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
        if (row.empty() || malformed || isSkippableControlRow(row)) continue;
        reconcileRowWidth(row);
        repairDateShiftedRow(row);
        bool accepted = true;
        if (decisionIdx < acceptedRowMask.size()) {
            accepted = (acceptedRowMask[decisionIdx] != 0);
        } else {
            accepted = !isLikelyMetadataRow(row, runningSums, r);
        }
        ++decisionIdx;
        if (!accepted) continue;
        if (r >= rowCount_) break;

        size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < header.size(); ++c) {
            std::string s = (c < cols) ? row[c] : "";
            if (isMissingToken(s)) {
                columns_[c].missing[r] = static_cast<uint8_t>(1);
                continue;
            }

            if (columns_[c].type == ColumnType::NUMERIC) {
                auto& numericValues = std::get<std::vector<double>>(columns_[c].values);
                double dv = 0.0;
                if (parseDouble(s, dv)) {
                    numericValues[r] = dv;
                } else {
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            } else if (columns_[c].type == ColumnType::DATETIME) {
                auto& datetimeValues = std::get<std::vector<int64_t>>(columns_[c].values);
                int64_t ts = 0;
                datetimeObserved[c]++;
                if (parseDateTime(s, ts)) {
                    datetimeValues[r] = ts;
                } else {
                    datetimeParseFailures[c]++;
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            } else {
                auto& categoricalValues = std::get<std::vector<std::string>>(columns_[c].values);
                categoricalValues[r] = CommonUtils::trim(s);
                if (categoricalValues[r].empty()) {
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            }
        }
        updateRunningSums(row, runningSums);
        ++r;
    }

    constexpr double kDatetimeFallbackFailureRatio = 0.15;
    constexpr double kDateHeaderFallbackFailureRatio = 0.40;
    std::vector<size_t> datetimeFallbackCols;
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (forcedTypeMask[c]) continue;
        if (columns_[c].type != ColumnType::DATETIME) continue;
        if (datetimeObserved[c] == 0 || datetimeParseFailures[c] == 0) continue;

        const double failureRatio = static_cast<double>(datetimeParseFailures[c]) /
                                    static_cast<double>(datetimeObserved[c]);
        const double allowedFailureRatio = dateLikeHeaderMask[c]
            ? kDateHeaderFallbackFailureRatio
            : kDatetimeFallbackFailureRatio;
        if (failureRatio <= allowedFailureRatio) continue;

        datetimeFallbackCols.push_back(c);

        columns_[c].type = ColumnType::CATEGORICAL;
        columns_[c].values = std::vector<std::string>(rowCount_);
        columns_[c].missing.assign(rowCount_, static_cast<uint8_t>(1));
    }

    if (datetimeFallbackCols.empty()) {
        return;
    }

    std::vector<uint8_t> fallbackMask(columns_.size(), static_cast<uint8_t>(0));
    for (size_t idx : datetimeFallbackCols) {
        fallbackMask[idx] = static_cast<uint8_t>(1);
    }

    in.clear();
    in.seekg(0, std::ios::beg);
    CSVUtils::skipBOM(in);
    malformed = false;
    auto headerThirdPass = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
    if (malformed || headerThirdPass.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");

    size_t rowIdx = 0;
    size_t fallbackDecisionIdx = 0;
    std::fill(runningSums.begin(), runningSums.end(), 0.0);
    while (in.peek() != EOF) {
        auto row = CSVUtils::parseCSVLine(in, delimiter_, &malformed, nullptr);
        if (row.empty() || malformed || isSkippableControlRow(row)) continue;
        reconcileRowWidth(row);
        repairDateShiftedRow(row);
        bool accepted = true;
        if (fallbackDecisionIdx < acceptedRowMask.size()) {
            accepted = (acceptedRowMask[fallbackDecisionIdx] != 0);
        } else {
            accepted = !isLikelyMetadataRow(row, runningSums, rowIdx);
        }
        ++fallbackDecisionIdx;
        if (!accepted) continue;
        if (rowIdx >= rowCount_) break;

        const size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < header.size(); ++c) {
            if (!fallbackMask[c]) continue;
            auto& categoricalValues = std::get<std::vector<std::string>>(columns_[c].values);
            std::string s = (c < cols) ? CommonUtils::trim(row[c]) : std::string();
            if (isMissingToken(s) || s.empty()) {
                columns_[c].missing[rowIdx] = static_cast<uint8_t>(1);
                categoricalValues[rowIdx].clear();
            } else {
                columns_[c].missing[rowIdx] = static_cast<uint8_t>(0);
                categoricalValues[rowIdx] = s;
            }
        }
        updateRunningSums(row, runningSums);
        ++rowIdx;
    }

    std::vector<size_t> rescuedDatetimeCols;
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (forcedTypeMask[c]) continue;
        if (!dateLikeHeaderMask[c]) continue;
        if (columns_[c].type != ColumnType::CATEGORICAL) continue;

        const auto& catVals = std::get<std::vector<std::string>>(columns_[c].values);
        size_t observed = 0;
        size_t datetimeLikeHits = 0;
        size_t epochHits = 0;
        for (size_t rIdx = 0; rIdx < catVals.size(); ++rIdx) {
            if (columns_[c].missing[rIdx]) continue;
            ++observed;
            int64_t ts = 0;
            if (parseDateTime(catVals[rIdx], ts)) {
                ++datetimeLikeHits;
            }
            if (parseLikelyEpochSeconds(catVals[rIdx], ts)) {
                ++epochHits;
            }
        }

        if (observed == 0) continue;
        const bool rescueByDatetime = datetimeLikeHits >= std::max<size_t>(3, (observed * 6) / 10);
        const bool rescueByEpoch = epochHits >= std::max<size_t>(2, (observed * 2) / 10);
        if (!rescueByDatetime && !rescueByEpoch) continue;

        columns_[c].type = ColumnType::DATETIME;
        columns_[c].values = std::vector<int64_t>(rowCount_, 0);
        auto& datetimeVals = std::get<std::vector<int64_t>>(columns_[c].values);

        for (size_t rIdx = 0; rIdx < catVals.size() && rIdx < rowCount_; ++rIdx) {
            if (columns_[c].missing[rIdx]) continue;
            int64_t ts = 0;
            if (parseDateTime(catVals[rIdx], ts)) {
                datetimeVals[rIdx] = ts;
                columns_[c].missing[rIdx] = static_cast<uint8_t>(0);
            } else {
                columns_[c].missing[rIdx] = static_cast<uint8_t>(1);
            }
        }
        rescuedDatetimeCols.push_back(c);
    }

}

std::vector<size_t> TypedDataset::numericColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::NUMERIC) out.push_back(i);
    return out;
}

std::vector<size_t> TypedDataset::categoricalColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::CATEGORICAL) out.push_back(i);
    return out;
}

std::vector<size_t> TypedDataset::datetimeColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::DATETIME) out.push_back(i);
    return out;
}

int TypedDataset::findColumnIndex(const std::string& name) const {
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].name == name) return static_cast<int>(i);
    return -1;
}

void TypedDataset::removeRows(const MissingMask& keepMask) {
    if (keepMask.size() != rowCount_) throw Seldon::DatasetException("Row mask size mismatch");

    for (auto& col : columns_) {
        MissingMask newMissing;
        newMissing.reserve(rowCount_);

        if (col.type == ColumnType::NUMERIC) {
            auto& values = std::get<std::vector<double>>(col.values);
            std::vector<double> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        } else if (col.type == ColumnType::CATEGORICAL) {
            auto& values = std::get<std::vector<std::string>>(col.values);
            std::vector<std::string> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        } else {
            auto& values = std::get<std::vector<int64_t>>(col.values);
            std::vector<int64_t> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        }

        col.missing = std::move(newMissing);
    }

    rowCount_ = std::count(keepMask.begin(), keepMask.end(), static_cast<uint8_t>(1));
}
