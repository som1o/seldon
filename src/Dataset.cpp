#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

Dataset::Dataset(const std::string& filename) : filename_(filename) {}

bool Dataset::load() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename_ << "\n";
        return false;
    }

    std::string line;
    // Read header
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            // Trim whitespace
            col.erase(col.find_last_not_of(" \n\r\t") + 1);
            col.erase(0, col.find_first_not_of(" \n\r\t"));
            columnNames.push_back(col);
        }
    }

    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        bool validRow = true;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (const std::exception&) {
                // Ignore non-numerical for now
                row.push_back(0.0); 
                validRow = false; // Could optionally reject the row
            }
        }
        if (!row.empty() && row.size() == columnNames.size()) {
            data.push_back(row);
        }
    }

    return true;
}

void Dataset::printSummary() const {
    std::cout << "Dataset Loaded: " << filename_ << "\n";
    std::cout << "Rows: " << getRowCount() << ", Columns: " << getColCount() << "\n";
}
