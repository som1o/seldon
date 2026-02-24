#pragma once

#include <istream>
#include <string>
#include <vector>

namespace CSVUtils {
// Low-level CSV tokenization and header normalization utilities.
// This module does not infer semantic types.
std::string trimUnquotedField(const std::string& value);
void skipBOM(std::istream& is);
std::vector<std::string> parseCSVLine(std::istream& is, char delimiter, bool* malformed = nullptr, size_t* consumedLines = nullptr);
std::vector<std::string> normalizeHeader(const std::vector<std::string>& header);
}
