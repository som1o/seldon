#pragma once

#include <istream>
#include <string>
#include <vector>

namespace CSVUtils {
std::string trimUnquotedField(const std::string& value);
void skipBOM(std::istream& is);
std::vector<std::string> parseCSVLine(std::istream& is, char delimiter, bool* malformed = nullptr, size_t* consumedLines = nullptr);
std::vector<std::string> normalizeHeader(const std::vector<std::string>& header);
}
