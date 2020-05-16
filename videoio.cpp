#include "videoio.hpp"
#include <iostream>


FileIO::FileIO(std::string video_in)
    : capture(video_in), filename(video_in)
{
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open video stream '" + video_in + "'");

    size_t lastdot = filename.find_last_of(".");
    std::string txt = (lastdot == std::string::npos) ? filename : filename.substr(0, lastdot);
    txt += ".txt";

    rect_file.open(txt);
    if (!rect_file.is_open())
        std::cout << txt << " not found - using empty init rectangle" << std::endl;
}

FileIO::FileIO(int camera_idx)
    : capture(camera_idx)
{
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open camera " + std::to_string(camera_idx));
}

cv::Rect FileIO::getInitRectangle()
{
    if (!rect_file.is_open())
        return cv::Rect();

    if (rect_line.empty())
        std::getline(rect_file, rect_line);

    cv::Rect r;
    if (sscanf(rect_line.c_str(), "%d,%d,%d,%d", &r.x, &r.y, &r.width, &r.height) == 4) {
        rect_line.clear();
        return r;
    }
    int num;
    if (sscanf(rect_line.c_str(), "%d:%d,%d,%d,%d", &num, &r.x, &r.y, &r.width, &r.height) == 5) {
        if (num == getImageNum()) {
            rect_line.clear();
            return r;
        } else {
            return cv::Rect();
        }
    }
    std::cerr << "Error parsing init rectangle: " << rect_line << std::endl;
    rect_line.clear();
    return cv::Rect();
}

void FileIO::outputBoundingBox(const cv::Rect &bbox)
{
    (void)bbox;
}

int FileIO::getNextFileName(char *fName)
{
    (void)fName;
    return 0;
}

int FileIO::getNextImage(cv::Mat &img)
{
    capture >> img;
    num++;
    return img.empty() ? 0 : 1;
}

int FileIO::getNextImage(cv::UMat &img)
{
    capture >> img;
    num++;
    return img.empty() ? 0 : 1;
}

int FileIO::getImageNum() const
{
    return num;
}
