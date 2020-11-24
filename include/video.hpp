#ifndef VIDEO_HPP
#define VIDEO_HPP

#include <iostream>
#include <vector>
#include <string>

class Video
{  
    public:
    bool finished;
    std::string filepath;
    std::vector<int> detection_framenums;

    // Member functions
    Video();
    Video(std::string &video_path);
};

#endif