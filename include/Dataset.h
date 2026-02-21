#pragma once
#include <string>
#include <vector>

class Dataset {
public:
    Dataset(const std::string& filename);

    // skipMalformed=true will silently discard unparseable rows and warn.
    // skipMalformed=false will abort loading on first bad row.
    bool load(bool skipMalformed = true);
    void printSummary() const;

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
};
