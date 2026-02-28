#include "AutoConfig.h"
#include "AutomationPipeline.h"
#include "SeldonGui.h"
#include "SeldonExceptions.h"

#include <cstdlib>
#include <iostream>
#include <thread>
#ifdef USE_OPENMP
#include <omp.h>
#endif

int main(int argc, char* argv[]) {
    try {
#ifdef USE_OPENMP
        if (std::getenv("OMP_NUM_THREADS") == nullptr) {
            const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
            omp_set_num_threads(static_cast<int>(hw));
            omp_set_dynamic(0);
        }
#endif
#ifdef SELDON_ENABLE_GUI
        if (argc > 1 && std::string(argv[1]) == "--cli") {
            AutoConfig config = AutoConfig::fromArgs(argc - 1, argv + 1);
            AutomationPipeline pipeline;
            return pipeline.run(config);
        }

        SeldonGui gui;
        return gui.run(argc, argv);
#else
        AutoConfig config = AutoConfig::fromArgs(argc, argv);
        AutomationPipeline pipeline;
        return pipeline.run(config);
#endif
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
