#pragma once

#include "AutoConfig.h"

#include <functional>
#include <string>

class AutomationPipeline final {
public:
    int run(const AutoConfig& config);

    /**
     * Optional progress hook for external consumers (e.g., web streaming).
     * Set before calling run().
     * Called on the pipeline thread with (label, step, totalSteps).
     */
    static std::function<void(const std::string&, int, int)> onProgress;
};
