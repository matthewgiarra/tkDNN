#ifndef TRIMMERIO_HPP
#define TRIMMERIO_HPP

#include <string>
#include "video.hpp"

int write_results(Video &video, std::string config_filepath);
int write_video(Video &video, std::string output_filepath);


#endif