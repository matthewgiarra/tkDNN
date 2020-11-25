#include "trimmerio.hpp"
#include "video.hpp"
#include "nlohmann/json.hpp"
// #include <string>
#include <iostream>
#include <queue>
#include "boost/filesystem.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/videoio.hpp"
#include "constants.hpp"

int write_results(Video &video, std::string config_filepath)
{
    // Read the config file
    std::ifstream input_stream;
    input_stream.open(config_filepath);
    if (!input_stream.is_open())
    {
        std::cerr << "Could not open input file invalid path.\tfilepath: " << config_filepath << std::endl;
        return -ENOENT;
    }
    nlohmann::json config_data = nlohmann::json::parse(input_stream);
    input_stream.close();

    // Options
    bool config_write_video = config_data[g_options][g_write_videos];

    if(config_write_video)
    {
        // Get parameters
        boost::filesystem::path input_path(video.path);
        boost::filesystem::path output_file_name(input_path.stem().string() + config_data[g_files][g_output_video_suffix].get<std::string>() + input_path.extension().string());
        
        // Choose the output directory
        // If consolidating video outputs, put everything in output_dir_container
        // Otherwise put the output video next to the input video with suffix appended before extension 
        boost::filesystem::path output_dir_container;
        if(config_data[g_options][g_consolidate_videos]){
            output_dir_container = config_data[g_files][g_output_dir_container].get<std::string>();
        }
        else
        {
            output_dir_container = input_path.parent_path();
        }

        // Create the output directory
        boost::filesystem::create_directories(output_dir_container);

        // Construct the output file path (container path)
        boost::filesystem::path output_file_container = output_dir_container / output_file_name;

        // Now write the video
        write_video(video, output_file_container.string());
    }
}

int write_video(Video &video, std::string output_filepath)
{
    
    // Open the file for reading
    cv::VideoCapture cap(video.path);
    if(!cap.isOpened())
    {
        std::cerr << "Failed to open video file: " << video.path << std::endl;
        return(-1); //TODO: More descriptive return for error tracking
    }
    std::cout<< "Opened video file: " << video.path << std::endl;
    
    // Get some parameters from the input video
    int image_width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int image_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fourcc = cap.get(cv::CAP_PROP_FOURCC);
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter resultVideo;
    resultVideo.open(output_filepath, fourcc, fps, cv::Size(image_width, image_height));

    // Frame
    cv::Mat frame;

    int frame_num = 0;
    int frames_written = 0;
    while(!video.detection_framenums.empty())
    {
        cap >> frame;
        if(!frame.data)
        {
            break;
        }
        if(frame_num == video.detection_framenums.front())
        {
            // Add the frame to the video
            resultVideo << frame;

            // Pop the element from the queue
            video.detection_framenums.pop();

            // Increment the number of frames written
            frames_written++;
        }
        // Increment frame number
        frame_num += 1;
    }
    
    std::cout << "Wrote " << frames_written << " frames to " << "output_filepath" << std::endl;

}

