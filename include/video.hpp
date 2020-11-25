#ifndef VIDEO_HPP
#define VIDEO_HPP

#include <iostream>
#include <vector>
#include <string>
#include "boost/filesystem.hpp"

class Video
{  
    public:
    bool finished;
    std::string path;
    std::vector<int> detection_framenums;

    // Member functions
    Video();
    Video(std::string &video_path);
    Video(boost::filesystem::path &video_path);
};

#endif