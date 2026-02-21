#pragma once
#include <string>
#include <vector>

class Dataset {
public:
    Dataset(const std::string& filename);

    bool load();
    void printSummary() const;

    const std::vector<std::string>& getColumnNames() const { return columnNames; }
    const std::vector<std::vector<double>>& getData() const { return data; }
    size_t getRowCount() const { return data.empty() ? 0 : data.size(); }
    size_t getColCount() const { return columnNames.size(); }

private:
    std::string filename_;
    std::vector<std::string> columnNames;
    std::vector<std::vector<double>> data; // rows of numerical data
};
