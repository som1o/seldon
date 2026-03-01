#include "AutoConfig.h"
#include "AutomationPipeline.h"
#include "PredictionService.h"
#include "SeldonExceptions.h"
#include "WebDashboard.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
void printServiceUsage(const char* executable) {
    std::cout << "Usage: " << executable << " --serve --registry <path> [--host 0.0.0.0] [--port 8080] [--threads 8]\n";
}

void printWebUsage(const char* executable) {
    std::cout << "Usage: " << executable << " --web [--host 0.0.0.0] [--port 8090] [--ws-port 8091] [--threads 8]\n";
}

PredictionService::Config parseServiceConfig(int argc,
                                             char* argv[],
                                             std::string& registryPath) {
    PredictionService::Config config;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--registry" && i + 1 < argc) {
            registryPath = argv[++i];
            continue;
        }
        if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
            continue;
        }
        if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            const int parsed = std::stoi(argv[++i]);
            config.threadCount = static_cast<size_t>(std::max(1, parsed));
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            printServiceUsage(argv[0]);
            std::exit(0);
        }
        throw Seldon::ConfigurationException("Unknown or incomplete service argument: " + arg);
    }

    if (registryPath.empty()) {
        throw Seldon::ConfigurationException("Missing required --registry <path> for --serve mode");
    }
    if (config.port <= 0 || config.port > 65535) {
        throw Seldon::ConfigurationException("--port must be between 1 and 65535");
    }

    return config;
}

WebDashboard::Config parseWebConfig(int argc, char* argv[]) {
    WebDashboard::Config config;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
            continue;
        }
        if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--ws-port" && i + 1 < argc) {
            config.wsPort = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            const int parsed = std::stoi(argv[++i]);
            config.threads = static_cast<size_t>(std::max(1, parsed));
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            printWebUsage(argv[0]);
            std::exit(0);
        }
        throw Seldon::ConfigurationException("Unknown or incomplete web argument: " + arg);
    }

    if (config.port <= 0 || config.port > 65535 || config.wsPort <= 0 || config.wsPort > 65535) {
        throw Seldon::ConfigurationException("--port and --ws-port must be between 1 and 65535");
    }
    return config;
}
}

int main(int argc, char* argv[]) {
    try {
#ifdef USE_OPENMP
        if (std::getenv("OMP_NUM_THREADS") == nullptr) {
            const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
            omp_set_num_threads(static_cast<int>(hw));
            omp_set_dynamic(0);
        }
#endif

        if (argc > 1 && std::string(argv[1]) == "--serve") {
            std::string registryPath;
            const PredictionService::Config serviceConfig = parseServiceConfig(argc, argv, registryPath);
            ModelRegistry registry;
            registry.loadFromFile(registryPath);
            RequestMonitor monitor;
            PredictionService service(registry, monitor);
            return service.start(serviceConfig);
        }

        if (argc > 1 && std::string(argv[1]) == "--web") {
            const WebDashboard::Config webConfig = parseWebConfig(argc, argv);
            WebDashboard dashboard;
            return dashboard.start(webConfig);
        }

        if (argc > 1 && std::string(argv[1]) == "--cli") {
            AutoConfig config = AutoConfig::fromArgs(argc - 1, argv + 1);
            AutomationPipeline pipeline;
            return pipeline.run(config);
        }

        AutoConfig config = AutoConfig::fromArgs(argc, argv);
        AutomationPipeline pipeline;
        return pipeline.run(config);
    } catch (const Seldon::SeldonException& e) {
        std::cerr << "[Seldon][Error] " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Seldon][Error] Unexpected runtime failure: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[Seldon][Error] Unknown non-standard exception encountered.\n";
        return 1;
    }
}
