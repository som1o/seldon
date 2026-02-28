#pragma once

#include <istream>
#include <string>
#include <vector>

namespace CSVUtils {
// Low-level CSV tokenization and header normalization utilities.
// This module does not infer semantic types.
struct ParseLimits {
	size_t maxFieldBytes = 8 * 1024 * 1024;           // 8 MiB
	size_t maxRecordBytes = 64 * 1024 * 1024;         // 64 MiB
	size_t maxColumns = 20000;
	size_t maxPhysicalLinesPerRecord = 10000;
};

std::string trimUnquotedField(const std::string& value);
void skipBOM(std::istream& is);
std::vector<std::string> parseCSVLine(std::istream& is,
									  char delimiter,
									  bool* malformed = nullptr,
									  size_t* consumedLines = nullptr,
									  bool* limitExceeded = nullptr,
									  const ParseLimits& limits = ParseLimits{});
std::vector<std::string> normalizeHeader(const std::vector<std::string>& header);

class CSVChunkReader {
public:
	explicit CSVChunkReader(std::istream& is, char delimiter, ParseLimits limits = ParseLimits{});
	std::vector<std::vector<std::string>> readChunk(size_t maxRows,
											 bool* malformed = nullptr,
											 bool* limitExceeded = nullptr);

private:
	std::istream& is_;
	char delimiter_;
	ParseLimits limits_;
};
}
