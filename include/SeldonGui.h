#pragma once

#include "AutoConfig.h"

#include <string>
#include <vector>

class AutomationPipeline;

class SeldonGui final {
public:
    int run(int argc, char** argv);

private:
    struct AdvancedKeyValue {
        std::string key;
        std::string value;
    };

    static std::vector<std::string> splitArgs(const std::string& text);
    static std::string toBool(bool value);
};
