#pragma once

#include "AutoConfig.h"

#include <functional>
#include <string>

class AutomationPipeline final {
public:
    int run(const AutoConfig& config);

    /**
     * Optional GUI progress hook. Set before calling run().
     * Called on the pipeline thread with (label, step, totalSteps).
     * The GUI must marshal the call to the main thread itself.
     */
    static std::function<void(const std::string&, int, int)> onProgress;
};
