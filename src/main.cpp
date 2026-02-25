#include "AutoConfig.h"
#include "AutomationPipeline.h"
#include "SeldonExceptions.h"

#include <iostream>

int main(int argc, char* argv[]) {
    try {
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
