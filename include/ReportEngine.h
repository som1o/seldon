#pragma once
#include <string>
#include <vector>

class ReportEngine {
public:
    void addTitle(const std::string& title);
    void addParagraph(const std::string& text);
    void addTable(const std::string& title, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows);
    void addImage(const std::string& title, const std::string& imagePath);
    void save(const std::string& filePath) const;

private:
    static std::string escapeHtml(const std::string& text);
    std::string body_;
};
