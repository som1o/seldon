#include "AutomationPipeline.h"

#include <stdexcept>

// Define the static progress-callback
thread_local std::function<void(const std::string&, int, int)> AutomationPipeline::onProgress;
thread_local std::function<bool()> AutomationPipeline::shouldCancel;

#include "pipeline_parts/PipelineUnivariate.cpp"
#include "pipeline_parts/PipelineModeling.cpp"
#include "pipeline_parts/PipelineBivariate.cpp"
#include "pipeline_parts/PipelineReporting.cpp"
#include "pipeline_parts/PipelineRuntime.cpp"
