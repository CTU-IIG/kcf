#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <unistd.h>

#include "kcf.h"
#include "vot.hpp"

double calcAccuracy(std::string line, cv::Rect bb_rect, cv::Rect &groundtruth_rect)
{
    std::vector<float> numbers;
    std::istringstream s( line );
    float x;
    char ch;

    while (s >> x){
        numbers.push_back(x);
        s >> ch;
    }
    double x1 = std::min(numbers[0], std::min(numbers[2], std::min(numbers[4], numbers[6])));
    double x2 = std::max(numbers[0], std::max(numbers[2], std::max(numbers[4], numbers[6])));
    double y1 = std::min(numbers[1], std::min(numbers[3], std::min(numbers[5], numbers[7])));
    double y2 = std::max(numbers[1], std::max(numbers[3], std::max(numbers[5], numbers[7])));

    groundtruth_rect = cv::Rect(x1, y1, x2-x1, y2-y1);

    double rects_intersection = (groundtruth_rect & bb_rect).area();
    double rects_union = (groundtruth_rect | bb_rect).area();
    double accuracy = rects_intersection/rects_union;

    return accuracy;
}

int main(int argc, char *argv[])
{
    //load region, images and prepare for output
    std::string region, images, output;
    int visualize_delay = -1, fit_size_x = -1, fit_size_y = -1;
    KCF_Tracker tracker;

    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"debug",     no_argument,       0,  'd' },
            {"help",      no_argument,       0,  'h' },
            {"output",    required_argument, 0,  'o' },
            {"visualize", optional_argument, 0,  'v' },
            {"fit",       optional_argument, 0,  'f' },
            {0,           0,                 0,  0 }
        };

        int c = getopt_long(argc, argv, "dhv::f::o:",
                        long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'd':
            tracker.m_debug = true;
            break;
        case 'h':
            std::cerr << "Usage: \n"
                      << argv[0] << " [options]\n"
                      << argv[0] << " [options] <directory>\n"
                      << argv[0] << " [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]\n"
                      << "Options:\n"
                      << " --visualize | -v[delay_ms]\n"
                      << " --output    | -o <output.txt>\n"
                      << " --debug     | -d\n"
                      << " --fit       | -f[WxH]\n";
            exit(0);
            break;
        case 'o':
            output = optarg;
            break;
        case 'v':
            visualize_delay = optarg ? atol(optarg) : 1;
            break;
        case 'f':
            std::string sizes = optarg ? optarg : "128x128";
            std::string delimiter = "x";
            size_t pos = sizes.find(delimiter);
            std::string first_argument = sizes.substr(0, pos);
            sizes.erase(0, pos + delimiter.length());

            fit_size_x = stol(first_argument);
	    fit_size_y = stol(sizes);
            break;
        }
    }

    switch (argc - optind) {
    case 1:
        if (chdir(argv[optind]) == -1) {
            perror(argv[optind]);
            exit(1);
        }
        // Fall through
    case 0:
        region = access("groundtruth.txt", F_OK) == 0 ? "groundtruth.txt" : "region.txt";
        images = "images.txt";
        if (output.empty())
            output = "output.txt";
        break;
    case 2:
        // Fall through
    case 3:
        region = std::string(argv[optind + 0]);
        images = std::string(argv[optind + 1]);
        if (output.empty()) {
            if ((argc - optind) == 3)
                output = std::string(argv[optind + 2]);
            else
                output = std::string(dirname(argv[optind + 0])) + "/output.txt";
        }
        break;
    default:
        std::cerr << "Too many arguments\n";
        return 1;
    }
    VOT vot_io(region, images, output);

    // if groundtruth.txt is used use intersection over union (IOU) to calculate tracker accuracy
    std::ifstream groundtruth_stream;
    if (region.compare("groundtruth.txt") == 0) {
        groundtruth_stream.open(region.c_str());
        std::string line;
        std::getline(groundtruth_stream, line);
    }

    cv::Mat image;

    //img = firts frame, initPos = initial position in the first frame
    cv::Rect init_rect = vot_io.getInitRectangle();
    vot_io.outputBoundingBox(init_rect);
    vot_io.getNextImage(image);

    tracker.init(image, init_rect, fit_size_x, fit_size_y);

    BBox_c bb;
    cv::Rect bb_rect;
    double avg_time = 0., sum_accuracy = 0.;
    int frames = 0;
    while (vot_io.getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
         std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame";
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;

        bb = tracker.getBBox();
        bb_rect = cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h);
        vot_io.outputBoundingBox(bb_rect);

        if (groundtruth_stream.is_open()) {
            std::string line;
            std::getline(groundtruth_stream, line);

            cv::Rect groundtruthRect;
            double accuracy = calcAccuracy(line, bb_rect, groundtruthRect);
            if (visualize_delay >= 0)
                cv::rectangle(image, groundtruthRect, CV_RGB(255, 0,0), 1);
            std::cout << ", accuracy: " << accuracy;
            sum_accuracy += accuracy;
        }

        std::cout << std::endl;

        if (visualize_delay >= 0) {
            cv::rectangle(image, bb_rect, CV_RGB(0,255,0), 2);
            cv::imshow("output", image);
            int ret = cv::waitKey(visualize_delay);
            if (visualize_delay > 0 && ret != -1 && ret != 255)
                break;
        }

//        std::stringstream s;
//        std::string ss;
//        int countTmp = frames;
//        s << "imgs" << "/img" << (countTmp/10000);
//        countTmp = countTmp%10000;
//        s << (countTmp/1000);
//        countTmp = countTmp%1000;
//        s << (countTmp/100);
//        countTmp = countTmp%100;
//        s << (countTmp/10);
//        countTmp = countTmp%10;
//        s << (countTmp);
//        s << ".jpg";
//        s >> ss;
//        //set image output parameters
//        std::vector<int> compression_params;
//        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//        compression_params.push_back(90);
//        cv::imwrite(ss.c_str(), image, compression_params);
    }

    std::cout << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;
    if (groundtruth_stream.is_open()) {
        std::cout << "Average accuracy: " << sum_accuracy/frames << std::endl;
        groundtruth_stream.close();
    }

    return EXIT_SUCCESS;
}
