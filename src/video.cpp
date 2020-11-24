#include "video.hpp"

Video::Video(std::string &input_file_path)
{
    finished = false;
    filepath = input_file_path;
    detection_framenums.clear();
};