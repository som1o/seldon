#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "SeldonExceptions.h"

class Dataset {
public:
    enum class ImputationStrategy { SKIP, ZERO, MEAN, MEDIAN };
    Dataset(const std::string& filename);

    /**
     * @brief Loads a CSV dataset into memory with high performance.
     * @param skipMalformed If true, discard irregular rows.
     * @param exhaustiveScan If true, scans all rows to detect column types.
     * @param strategy Imputation strategy for missing values.
     * @throws Seldon::DatasetException if loading fails.
     */
    void load(bool skipMalformed = true, bool exhaustiveScan = false, ImputationStrategy strategy = ImputationStrategy::ZERO);

    /**
     * @brief Prints a visual summary of the dataset to the terminal.
     */
    void printSummary() const;

    /**
     * @brief Loads a specific chunk of the dataset. Useful for streaming.
     * @param startRow Row index to start from.
     * @param numRows Number of rows to load.
     * @throws Seldon::DatasetException if chunk loading fails.
     */
    void loadChunk(size_t startRow, size_t numRows);

    /**
     * @brief Automatically chooses between full load and chunk loading based on file size.
     */
    void loadAuto(bool skipMalformed = true, bool exhaustiveScan = false, ImputationStrategy strategy = ImputationStrategy::ZERO);

    const std::vector<std::string>& getColumnNames() const { return columnNames; }
    
    // Returns data grouped by columns instead of rows for better cache locality and performance
    const std::vector<std::vector<double>>& getColumns() const { return columns; }
    
    size_t getRowCount() const { return rowCount; }
    size_t getColCount() const { return columnNames.size(); }

    // Streaming pipeline
    void openStream();
    bool fetchNextChunk(size_t chunkSize);
    void closeStream();
    bool isStreamOpen() const { return streamFile.is_open(); }
    void analyzeMetadata(); // New: pass 0 to detect types and names

private:
    std::string filename_;
    size_t rowCount = 0;
    std::vector<std::vector<double>> columns;
    std::vector<std::string> columnNames;
    std::vector<size_t> numericIndices;
    std::vector<double> imputationValues;

    std::ifstream streamFile;


    // Metadata cache
    std::vector<std::string> allColumnNames;
    std::vector<bool> isNumeric;

    // Helper methods for decomposition
    std::vector<std::string> parseCSVLine(std::istream& is);
    void skipBOM(std::istream& file);
    std::vector<std::string> readHeader(std::ifstream& file);
    void detectColumnTypes(std::ifstream& file, size_t expectedCols, bool exhaustiveScan, 
                           std::vector<bool>& isNumeric, std::vector<bool>& hasValue, 
                           std::vector<std::vector<double>>& rawData, 
                           bool skipMalformed, size_t& totalSkipped);
    void processImputations(const std::vector<std::vector<double>>& rawData, 
                            const std::vector<bool>& isNumeric, 
                            const std::vector<std::string>& allColumnNames, 
                            ImputationStrategy strategy);
    void calculateImputations(const std::vector<std::vector<double>>& rawNumericData, 
                               std::vector<double>& imputationValues, 
                               ImputationStrategy strategy);
};
