#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <unistd.h>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>
#include <err.h>

#include "kcf.h"
#include "vot.hpp"
#include "videoio.hpp"
#include <opencv2/core/core_c.h>

// Needed for OpenCV <= 3.2 as replacement for Rect::empty()
bool empty(cv::Rect r)
{
    return r.width <= 0 || r.height <= 0;
}

#if CV_MAJOR_VERSION < 3
void setWindowTitle(const cv::String& winname, const cv::String& title)
{
    (void)winname;
    (void)title;
}
#endif

void writeBBox(std::string box_out, int frame_num, cv::Rect b)
{
    // TODO: Check for errors
    FILE *f = fopen(box_out.c_str(), "a");
    fprintf(f, "%d:%d,%d,%d,%d\n", frame_num, b.x, b.y, b.width, b.height);
    fclose(f);
}

cv::Rect selectBBox(cv::Mat image, std::string box_out = "", int frame_num = 0)
{
    using namespace cv;

    struct state {
        Mat img;
        Rect bbox;
        bool secondPoint = false;
        bool done = false;
    } state;

    state.img = image;

    namedWindow("KCF output", WINDOW_NORMAL);
    setWindowTitle("KCF output", "Select region to be tracked");

    auto callback = [](int event, int x, int y, int flags, void* userdata)
    {
        (void)flags;
        auto state = reinterpret_cast<struct state*>(userdata);

        Mat ui;
        state->img.copyTo(ui);

        switch (event) {
        case EVENT_MOUSEMOVE:
            if (!state->secondPoint) {
                line(ui, Point(x, 0), Point(x, ui.rows), Scalar(0, 255, 0), 1);
                line(ui, Point(0, y), Point(ui.cols, y), Scalar(0, 255, 0), 1);
                state->bbox.x = x;
                state->bbox.y = y;
            } else {
                rectangle(ui, state->bbox.tl(), state->bbox.br(), Scalar(0, 255, 0), 1);
                state->bbox.width = x - state->bbox.x;
                state->bbox.height = y - state->bbox.y;
            }
            imshow("KCF output", ui);
            break;
        case EVENT_LBUTTONDOWN:
            if (!state->secondPoint) {
                state->secondPoint = true;
            } else {
                state->done = true;
            }
        }
    };

    callback(EVENT_MOUSEMOVE, image.cols / 2, image.rows / 2, 0, &state);
    setMouseCallback("KCF output", callback, &state);

    int key = 0;
    while (key != 27 /*esc*/ && key != 'q' && !state.done) {
        key = waitKey(50);
    }

    if (state.done && !box_out.empty())
        writeBBox(box_out, frame_num, state.bbox);

    setWindowTitle("KCF output", "KCF output");
    setMouseCallback("KCF output", nullptr);
    return state.bbox;
}

double calcAccuracy(std::string line, cv::Rect bb_rect, cv::Rect &groundtruth_rect)
{
    std::vector<float> numbers;
    std::istringstream s(line);
    float x;
    char ch;

    while (s >> x) {
        numbers.push_back(x);
        s >> ch;
    }
    double x1 = std::min(numbers[0], std::min(numbers[2], std::min(numbers[4], numbers[6])));
    double x2 = std::max(numbers[0], std::max(numbers[2], std::max(numbers[4], numbers[6])));
    double y1 = std::min(numbers[1], std::min(numbers[3], std::min(numbers[5], numbers[7])));
    double y2 = std::max(numbers[1], std::max(numbers[3], std::max(numbers[5], numbers[7])));

    groundtruth_rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);

    double rects_intersection = (groundtruth_rect & bb_rect).area();
    double rects_union = (groundtruth_rect | bb_rect).area();
    double accuracy = rects_intersection / rects_union;

    return accuracy;
}

int main(int argc, char *argv[])
{
    //load region, images and prepare for output
    std::string region, images, output, video_out, box_out;
    int visualize_delay = -1, fit_size_x = -1, fit_size_y = -1;
    KCF_Tracker tracker;
    cv::VideoWriter videoWriter;
    cv::Rect init_rect;
    bool set_box_interactively = false;
    bool do_track = true;

    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"debug",     no_argument,       0,  'd' },
            {"visual_debug", optional_argument,    0, 'p'},
            {"help",      no_argument,       0,  'h' },
            {"output",    required_argument, 0,  'o' },
            {"video_out", optional_argument, 0,  'O' },
            {"visualize", optional_argument, 0,  'v' },
            {"fit",       optional_argument, 0,  'f' },
            {"box",       optional_argument, 0,  'b' },
            {"box_out",   required_argument, 0,  'B' },
            {0,           0,                 0,  0 }
        };

        int c = getopt_long(argc, argv, "b::B:dp::hv::f::o:O::", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'b':
            if (optarg) {
                if (sscanf(optarg, "%d,%d,%d,%d", &init_rect.x, &init_rect.y, &init_rect.width, &init_rect.height) != 4)
                    errx(1, "Invalid box specification: %s", optarg);
            } else {
                set_box_interactively = true;
            }
            break;
        case 'B':
            box_out = optarg;
            break;
        case 'd':
            tracker.m_debug = true;
            break;
        case 'p':
            if (!optarg || *optarg == 'p')
                tracker.m_visual_debug = KCF_Tracker::vd::PATCH;
            else if (optarg && *optarg == 'r')
                tracker.m_visual_debug = KCF_Tracker::vd::RESPONSE;
            else {
                fprintf(stderr, "Unknown visual debug mode: %c", *optarg);
                return 1;
            }
            break;
        case 'h':
            std::cerr << "Usage: \n"
                      << argv[0] << " [options]\n"
                      << argv[0] << " [options] <directory>\n"
                      << argv[0] << " [options] <video_file>\n"
                      << argv[0] << " [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]\n"
                      << "Options:\n"
                      << " --visualize    | -v[delay_ms]\n"
                      << " --output       | -o <output.txt>\n"
                      << " --video_out    | -O <filename>\n"
                      << " --fit          | -f[W[xH]]\n"
                      << " --debug        | -d\n"
                      << " --visual_debug | -p [p|r]\n"
                      << " --box          | -b [X,Y,W,H]\n"
                      << " --box_out      | -B <filename>\n";
            exit(0);
            break;
        case 'o':
            output = optarg;
            break;
        case 'O':
            video_out = optarg ? optarg : "./output.avi";
            break;
        case 'v':
            visualize_delay = optarg ? atol(optarg) : 1;
            break;
        case 'f':
            if (!optarg) {
                fit_size_x = fit_size_y = 0;
            } else {
                char tail;
                if (sscanf(optarg, "%d%c", &fit_size_x, &tail) == 1) {
                    fit_size_y = fit_size_x;
                } else if (sscanf(optarg, "%dx%d%c", &fit_size_x, &fit_size_y, &tail) != 2) {
                    fprintf(stderr, "Cannot parse -f argument: %s\n", optarg);
                    return 1;
                }
            }
            break;
        }
    }

    std::unique_ptr<VideoIO> io;

    switch (argc - optind) {
    case 1:
        try { // If the argument is a number, try openning the camera first
            io.reset(new FileIO(std::stoi(argv[optind])));
            break;
        }
        catch (std::exception& e) {
            // No number or camera error, continue
        }

        struct stat st;
        if (stat(argv[optind], &st) != 0) {
            perror(argv[optind]);
            exit(1);
        }
        if (S_ISDIR(st.st_mode)) {
            if (chdir(argv[optind]) == -1) {
                perror(argv[optind]);
                exit(1);
            }
        } else if (S_ISREG(st.st_mode)) {
            io.reset(new FileIO(argv[optind]));
            break;
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

    if (!io)
        io.reset(new VOT(region, images, output));

    // if groundtruth.txt is used use intersection over union (IOU) to calculate tracker accuracy
    std::ifstream groundtruth_stream;
    if (region.compare("groundtruth.txt") == 0) {
        groundtruth_stream.open(region.c_str());
        std::string line;
        std::getline(groundtruth_stream, line);
    }

    cv::Mat image;
    io->getNextImage(image);

    //img = firts frame, initPos = initial position in the first frame
    if (empty(init_rect))
        init_rect = io->getInitRectangle(); // Try to get BBox from VOT or .txt files

    if (empty(init_rect) || set_box_interactively) {
        init_rect = selectBBox(image, box_out, 1);
        auto b = init_rect;
        printf("--box=%d,%d,%d,%d\n", b.x, b.y, b.width, b.height);
        if (visualize_delay < 0)
            cv::destroyWindow("KCF output");
    }
    io->outputBoundingBox(init_rect);

    if (!video_out.empty()) {
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
        double fps = 25.0;                          // framerate of the created video stream
        videoWriter.open(video_out, codec, fps, image.size(), true);
    }

    tracker.init(image, init_rect, fit_size_x, fit_size_y);


    BBox_c bb;
    cv::Rect bb_rect;
    double avg_time = 0., sum_accuracy = 0.;
    int frames = 0;

    std::cout << std::fixed << std::setprecision(2);

    while (io->getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        init_rect = io->getInitRectangle();
        if (init_rect.x == -1)
            do_track = false;
        if (empty(init_rect) && do_track) {
            tracker.track(image);
        } else {
            do_track = true;
            tracker.init(image, init_rect, fit_size_x, fit_size_y);
        }

        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
        std::cout << io->getImageNum() << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms per frame, "
                      "response : " << tracker.getFilterResponse();
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;

        bb = tracker.getBBox();
        bb_rect = cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h);
        io->outputBoundingBox(bb_rect);

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

        if (visualize_delay >= 0 || !video_out.empty()) {
            cv::Point pt(bb.cx, bb.cy);
            cv::Size size(bb.w, bb.h);
            cv::RotatedRect rotatedRectangle(pt, size, bb.a);

            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);

            if (do_track)
                for (int i = 0; i < 4; i++)
                    cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            if (visualize_delay >= 0) {
                cv::imshow("KCF output", image);

                int key = cv::waitKey(visualize_delay);
                if (key == 27 /*esc*/ || key == 'q')
                    break;
                switch (key) {
                case 'i':
                    init_rect = selectBBox(image, box_out, io->getImageNum());
                    tracker.init(image, init_rect, fit_size_x, fit_size_y);
                    break;
                case 'o':
                    // switch tracker off
                    do_track = false;
                    writeBBox(box_out, io->getImageNum(), cv::Rect(-1,-1,-1,-1));
                    break;
                }
            }
            if (!video_out.empty())
                videoWriter << image;
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

    std::cout << "Average processing speed: " << avg_time / frames << "ms (" << 1. / (avg_time / frames) * 1000 << " fps)";
    if (groundtruth_stream.is_open()) {
        std::cout << "; Average accuracy: " << sum_accuracy/frames << std::endl;
        groundtruth_stream.close();
    }
    if (!video_out.empty())
       videoWriter.release();
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
