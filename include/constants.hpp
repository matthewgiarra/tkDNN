#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>

/**
 * These constants are the lookup strings for the setup .json file generated by python.
 * By making them constant variables it ensures if they ever change the strings only need to be updated here.
 */

// Classes
const std::string g_classes{"classes"};

// Files
const std::string g_files{"files"};
const std::string g_data_dir_host{"data_dir_host"};
const std::string g_data_dir_container{"data_dir_container"};
const std::string g_workspace_dir_container{"workspace_dir_container"};
const std::string g_video_list_path_workspace{"video_list_path_workspace"};
const std::string g_model_path_workspace{"model_path_workspace"};

// Options
const std::string g_options{"options"};
const std::string g_write_videos{"write_videos"};
const std::string g_consolidate_videos{"consolidate_videos"};
const std::string g_output_video_suffix{"output_video_suffix"};
const std::string g_output_dir_container{"output_dir_container"};

// Parameters
const std::string g_parameters{"parameters"};
const std::string g_confidence_threshold{"confidence_threshold"};
const std::string g_frame_step{"frame_step"};
const std::string g_batch_size{"batch_size"};

#endif