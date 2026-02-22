#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

enum class ColumnType { NUMERIC, CATEGORICAL, DATETIME };
using ColumnStorage = std::variant<std::vector<double>, std::vector<std::string>, std::vector<int64_t>>;

struct TypedColumn {
    std::string name;
    ColumnType type = ColumnType::CATEGORICAL;
    ColumnStorage values = std::vector<std::string>{};
    std::vector<bool> missing;
};

class TypedDataset {
public:
    explicit TypedDataset(std::string filename, char delimiter = ',');

    /**
     * @brief Loads CSV content and infers per-column types.
     * @pre file exists and is readable.
     * @post columns() is populated with aligned typed vectors and missing masks.
     * @throws Seldon::IOException / Seldon::DatasetException on IO or parse failure.
     */
    void load();
    size_t rowCount() const { return rowCount_; }
    size_t colCount() const { return columns_.size(); }

    const std::vector<TypedColumn>& columns() const { return columns_; }
    std::vector<TypedColumn>& columns() { return columns_; }

    std::vector<size_t> numericColumnIndices() const;
    std::vector<size_t> categoricalColumnIndices() const;
    std::vector<size_t> datetimeColumnIndices() const;

    /**
     * @brief Returns index of named column or -1 when absent.
     */
    int findColumnIndex(const std::string& name) const;

    /**
     * @brief Removes rows where keepMask is false across all columns.
     * @pre keepMask.size() == rowCount().
     * @post All typed columns keep row alignment after filtering.
     * @throws Seldon::DatasetException when mask size mismatches row count.
     */
    void removeRows(const std::vector<bool>& keepMask);

private:
    std::string filename_;
    char delimiter_;
    size_t rowCount_ = 0;
    std::vector<TypedColumn> columns_;

    std::vector<std::string> parseCSVLine(std::istream& is, bool& malformed, size_t& consumedLines) const;
    static bool parseDouble(const std::string& v, double& out);
    static bool parseDateTime(const std::string& v, int64_t& outUnixSeconds);
    static std::string trim(const std::string& s);
};
