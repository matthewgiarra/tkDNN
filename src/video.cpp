#include "video.hpp"
#include "boost/filesystem.hpp"

Video::Video(std::string &input_file_path)
{
    finished = false;
    path = input_file_path;
    
};

Video::Video(boost::filesystem::path &input_file_path)
{
    finished = false;
    path = input_file_path.string();
    
};