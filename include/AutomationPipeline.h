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
    static thread_local std::function<void(const std::string&, int, int)> onProgress;

    /**
     * Optional cancellation hook for external consumers (e.g., web UI cancel).
     * Should return true when the current run should stop as soon as possible.
     */
    static thread_local std::function<bool()> shouldCancel;
};
