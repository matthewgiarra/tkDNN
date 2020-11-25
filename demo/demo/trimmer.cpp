#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <fstream>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"
#include "nlohmann/json.hpp"
#include "constants.hpp"
#include "video.hpp"
#include "boost/filesystem.hpp"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[])
{

    std::cout << "HELLO DINGUS" << std::endl;
    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);

    // Print out the input arguments
    for(int i = 0; i < argc; i++)
    {
        std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
    }

    std::string net = "yolo3_berkeley.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 
    char ntype = 'y';
    if(argc > 3)
        ntype = argv[3][0]; 
    int n_classes = 80;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    int n_batch = 1;
    if(argc > 5)
        n_batch = atoi(argv[5]); 
    bool show = true;
    if(argc > 6)
        show = atoi(argv[6]); 
    float conf_thresh=0.3;
    if(argc > 7)
        conf_thresh = atof(argv[7]);
    std::string config_filepath=std::string("config.json");
    if(argc > 8)
        config_filepath = std::string(argv[8]);

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    // Print the path to the json file that will be parsed.
    std::cout << "Trimmer config filepath: " << config_filepath << std::endl;
    
    // Open and parse the json config file
    std::ifstream input_stream;
    input_stream.open(config_filepath);
    if (!input_stream.is_open())
    {
        std::cerr << "Could not open input file invalid path.\tfilepath: " << config_filepath << std::endl;
        return -ENOENT;
    }
    nlohmann::json config_data = nlohmann::json::parse(input_stream);
    input_stream.close();

    // Path to the top-level data directory
    boost::filesystem::path data_dir_host             = config_data[g_files][g_data_dir_host];
    boost::filesystem::path data_dir_container        = config_data[g_files][g_data_dir_container];
    boost::filesystem::path workspace_dir_container   = config_data[g_files][g_workspace_dir_container];
    boost::filesystem::path video_list_path_workspace = config_data[g_files][g_video_list_path_workspace];
    boost::filesystem::path video_list_path_container = workspace_dir_container / video_list_path_workspace;

    if(!boost::filesystem::exists(video_list_path_container))
    {
        std::cerr << "Video list file not found: " << video_list_path_container.string() << std::endl;
        return -ENOENT;
    }
    else
    {
        std::cout << "Found video list file: " << video_list_path_container.string() << std::endl;
    }
    
    // Read the video list
    input_stream.open(video_list_path_container.string());
    if (!input_stream.is_open())
    {
        std::cerr << "Could not open input file invalid path.\tfilepath: " << video_list_path_container << std::endl;
        return -ENOENT;
    }
    nlohmann::json file_list_data = nlohmann::json::parse(input_stream);
    input_stream.close();

    // Turn the video list into a vector of boost filepaths
    std::vector<std::string> video_path_list_str = file_list_data[g_files];
    std::vector<boost::filesystem::path> video_path_list;
    std::vector<Video> video_obj_list;
    for(int i = 0; i < video_path_list_str.size(); i++)
    {
        // Absolute path to the file on the host
        boost::filesystem::path video_path_host_abs(video_path_list_str[i]);
        
        // Video path relative to the data directory on the host
        boost::filesystem::path video_path_host_rel = boost::filesystem::relative(video_path_host_abs, data_dir_host);

        // Absolute path to the video in the container
        // (made by joining relative host path to container data root path)
        boost::filesystem::path video_path_container = data_dir_container / video_path_host_rel;

        // Search for the file
        if(!boost::filesystem::exists(video_path_container))
        {
            std::cerr << "Video file not found:" <<std::endl;
            std::cerr << "Container path: " << video_path_container.string() << std::endl;
            std::cerr << "Host path: "      << video_path_host_abs.string() << std::endl;
            return -ENOENT;
        }
        else
        {
            // Add the video path (container path) to the vector of video paths  
            std::cout << "Adding video to list: " << video_path_host_abs.string() << std::endl;
            video_path_list.push_back(video_path_container);
            video_obj_list.push_back(Video(video_path_container));
        }
    }
    int num_videos = video_obj_list.size();

    // Initialize the network
    detNN->init(net, n_classes, n_batch, conf_thresh);
    std::cout << "Classes in model: " << std::endl;
    for(int i = 0; i < n_classes; i++){
        std::cout << detNN->classesNames[i] << std::endl;
    }

    std::vector<std::string> trimmer_class_names = config_data[g_classes];
    // Get the class numbers of the class names specified in the config file
    std::vector<int> trimmer_class_nums(trimmer_class_names.size());
    for(int i = 0; i < trimmer_class_names.size(); i++){
        for(int j = 0; j < n_classes; j++){
            if(detNN->classesNames[j] == trimmer_class_names[i])
            {
                trimmer_class_nums[i] = j;
                break;
            }
        }
    }
    std::cout << "Trimming on classes:" << std::endl;
    for(int i = 0; i < trimmer_class_names.size(); i++){
        std::cout << trimmer_class_names[i] << " (" << trimmer_class_nums[i] << ")" << std::endl;
    }

    // Loop over the videos
    for(int vnum = 0; vnum < num_videos; vnum++)
    {
        
        // Image height and width
        int image_height, image_width;
        
        // Video path as a string
        std::string video_path = video_obj_list[vnum].path;
        gRun = true;
        cv::VideoCapture cap(video_path);
        if(!cap.isOpened())
        {
            std::cerr << "Failed to open video file: " << video_path << std::endl;
            gRun = false;
        }
        else
        {
            std::cout<<"Opened video file: " << video_path << std::endl;
            image_width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            image_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
            
        /*
        cv::VideoWriter resultVideo;
        if(SAVE_RESULT) {
            int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
        }
        */

        cv::Mat frame;
        if(show)
        {
            cv::namedWindow("detection", cv::WINDOW_NORMAL);
        }
            
        std::vector<cv::Mat> batch_frame;
        std::vector<cv::Mat> batch_dnn_input;
        tk::dnn::box bbox;
        std::string class_name;
        // std::vector<int> class_detected_frame_nums;
        
        // Number of batches processed
        int batch_num = 0;
        
        while(gRun)
        {
            batch_dnn_input.clear();
            batch_frame.clear();
            for(int bi=0; bi< n_batch; ++bi)
            {
                cap >> frame; 
                if(!frame.data)
                {
                    gRun = false;
                    break;
                } 
                    
                // Frame for drawing I think
                batch_frame.push_back(frame);

                // this will be resized to the net format
                batch_dnn_input.push_back(frame.clone());
            }
            
            if(batch_dnn_input.empty())
            {
                // No frames in batch; done with this video
                break;
            }

            // A few frames in this batch; need to pad with zeros
            // so the inference doesn't barf on an unexpected batch size
            if(batch_dnn_input.size() < n_batch)
            {
                int batch_deficit = n_batch - batch_dnn_input.size();
                for(int i = 0; i < batch_deficit; i++)
                {
                    batch_dnn_input.push_back(cv::Mat::zeros(image_height, image_width, CV_32F));
                }
            }
                
            // Do the inference
            detNN->update(batch_dnn_input, n_batch);
            detNN->draw(batch_frame);

            // Determine if any of the specified classes were detected
            // Loop over the frames in the batch
            for(int bi=0; bi<batch_frame.size(); ++bi)
            {
                // Initialize 'detected' value as false
                bool trimmer_classes_detected = false;

                // Loop over the detections for the bi'th frame in the batch
                for(int i=0; i < detNN->batchDetected[bi].size(); i++) { 

                    // Get info for the ith detection in the bi'th batch index
                    bbox = detNN->batchDetected[bi][i];
                    // Class number of detection
                    for(int j = 0; j < trimmer_class_nums.size(); j++)
                    {
                        if(trimmer_class_nums[j] == bbox.cl)
                        {
                            int frame_num = batch_num * n_batch + bi;
                            // class_detected_frame_nums.push_back(frame_num);
                            video_obj_list[vnum].detection_framenums.push_back(frame_num);
                            trimmer_classes_detected = true;
                            std::cout << video_path << ": detected " << detNN->classesNames[bbox.cl] << " in frame " << frame_num << std::endl;
                            break;
                        }
                    }

                    if(trimmer_classes_detected)
                    { 
                        break;
                    }
                }

                /*
                if(trimmer_classes_detected && SAVE_RESULT)
                {
                    resultVideo << batch_frame[bi];
                }
                */
            }

            if(show){
                for(int bi=0; bi< n_batch; ++bi){
                    cv::imshow("detection", batch_frame[bi]);
                    cv::waitKey(1);
                }
            }
            // if(n_batch == 1 && SAVE_RESULT)
                // resultVideo << frame;

            // Increment the number of processed batches
            batch_num++;     
        }

        std::cout << "Finished detections in " << video_path << std::endl;
    }
      
    std::cout<<"Done with batch\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    
    return 0;
}

