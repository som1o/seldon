#pragma once
#include <string>
#include <vector>

class Dataset {
public:
    enum class ImputationStrategy { SKIP, ZERO, MEAN, MEDIAN };
    Dataset(const std::string& filename);

    /**
     * @brief Loads a CSV dataset into memory with high performance.
     * @param skipMalformed If true, discard irregular rows.
     * @param exhaustiveScan If true, scans all rows to detect column types.
     * @param strategy Imputation strategy for missing values.
     * @return true if loading succeeded.
     */
    bool load(bool skipMalformed = true, bool exhaustiveScan = false, ImputationStrategy strategy = ImputationStrategy::ZERO);

    /**
     * @brief Prints a visual summary of the dataset to the terminal.
     */
    void printSummary() const;

    /**
     * @brief Loads a specific chunk of the dataset. Useful for streaming.
     * @param startRow Row index to start from.
     * @param numRows Number of rows to load.
     * @return true if chunk was loaded.
     */
    bool loadChunk(size_t startRow, size_t numRows);

    const std::vector<std::string>& getColumnNames() const { return columnNames; }
    
    // Returns data grouped by columns instead of rows for better cache locality and performance
    const std::vector<std::vector<double>>& getColumns() const { return columns; }
    
    size_t getRowCount() const { return rowCount; }
    size_t getColCount() const { return columnNames.size(); }

private:
    std::string filename_;
    std::vector<std::string> columnNames;
    std::vector<std::vector<double>> columns; // columns of numerical data
    size_t rowCount = 0;

    std::vector<size_t> numericIndices;
    std::vector<double> imputationValues;

    std::vector<std::string> parseCSVLine(std::istream& is);
    void calculateImputations(const std::vector<std::vector<double>>& rawNumericData, 
                               std::vector<double>& imputationValues, 
                               ImputationStrategy strategy);
};
