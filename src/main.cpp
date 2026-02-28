#include "AutoConfig.h"
#include "AutomationPipeline.h"
#include "SeldonGui.h"
#include "SeldonExceptions.h"

#include <iostream>

int main(int argc, char* argv[]) {
    try {
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
