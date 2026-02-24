#pragma once
#include <cstdint>
#include <unordered_map>
#include <string>
#include <variant>
#include <vector>

enum class ColumnType { NUMERIC, CATEGORICAL, DATETIME };
using ColumnStorage = std::variant<std::vector<double>, std::vector<std::string>, std::vector<int64_t>>;
using MissingMask = std::vector<uint8_t>;

struct TypedColumn {
    std::string name;
    ColumnType type = ColumnType::CATEGORICAL;
    ColumnStorage values = std::vector<std::string>{};
    MissingMask missing;
};

class TypedDataset {
public:
    enum class NumericSeparatorPolicy {
        AUTO,
        US_THOUSANDS,
        EUROPEAN
    };

    enum class DateLocaleHint {
        AUTO,
        DMY,
        MDY
    };

    explicit TypedDataset(std::string filename, char delimiter = ',');

    void setNumericSeparatorPolicy(NumericSeparatorPolicy policy) noexcept { numericSeparatorPolicy_ = policy; }
    void setDateLocaleHint(DateLocaleHint hint) noexcept { dateLocaleHint_ = hint; }
    void setColumnTypeOverride(std::string columnNameLower, ColumnType type) { columnTypeOverrides_[std::move(columnNameLower)] = type; }
    void setColumnTypeOverrides(std::unordered_map<std::string, ColumnType> overrides) { columnTypeOverrides_ = std::move(overrides); }

    /**
     * @brief Loads CSV content and infers per-column types.
        * @details CSV parsing/tokenization is delegated to CSVUtils; this class owns type inference and typed storage.
     * @pre file exists and is readable.
     * @post columns() is populated with aligned typed vectors and missing masks.
     * @throws Seldon::IOException / Seldon::DatasetException on IO or parse failure.
     */
    void load();
    size_t rowCount() const noexcept { return rowCount_; }
    size_t colCount() const noexcept { return columns_.size(); }

    const std::vector<TypedColumn>& columns() const noexcept { return columns_; }
    std::vector<TypedColumn>& columns() noexcept { return columns_; }

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
    void removeRows(const MissingMask& keepMask);

private:
    std::string filename_;
    char delimiter_;
    NumericSeparatorPolicy numericSeparatorPolicy_ = NumericSeparatorPolicy::AUTO;
    DateLocaleHint dateLocaleHint_ = DateLocaleHint::AUTO;
    std::unordered_map<std::string, ColumnType> columnTypeOverrides_;
    size_t rowCount_ = 0;
    std::vector<TypedColumn> columns_;

    bool parseDouble(const std::string& v, double& out) const;
    bool parseDateTime(const std::string& v, int64_t& outUnixSeconds) const;
};
