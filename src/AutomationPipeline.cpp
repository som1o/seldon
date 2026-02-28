#include "AutomationPipeline.h"

// Define the static progress-callback
std::function<void(const std::string&, int, int)> AutomationPipeline::onProgress;

#include "pipeline_parts/PipelineUnivariate.cpp"
#include "pipeline_parts/PipelineModeling.cpp"
#include "pipeline_parts/PipelineBivariate.cpp"
#include "pipeline_parts/PipelineReporting.cpp"
#include "pipeline_parts/PipelineRuntime.cpp"
