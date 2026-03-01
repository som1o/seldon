#pragma once

#include <cstddef>
#include <string>

class WebDashboard {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8090;
        int wsPort = 8091;
        size_t threads = 8;
    };

    int start(const Config& config);
};
