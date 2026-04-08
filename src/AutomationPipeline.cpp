#include "AutomationPipeline.h"

#include <stdexcept>

// Define the static progress-callback
thread_local std::function<void(const std::string&, int, int)> AutomationPipeline::onProgress;
thread_local std::function<bool()> AutomationPipeline::shouldCancel;
