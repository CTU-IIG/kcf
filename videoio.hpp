#ifndef VIDEOIO_HPP
#define VIDEOIO_HPP

#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>

class VideoIO {
public:
    virtual ~VideoIO() {}
    virtual cv::Rect getInitRectangle() = 0;
    virtual void outputBoundingBox(const cv::Rect & bbox) = 0;
    virtual int getNextFileName(char * fName) = 0;
    virtual int getNextImage(cv::UMat & img) = 0;
    virtual int getNextImage(cv::Mat & img) = 0;
    virtual int getImageNum() const = 0;
};

class FileIO : public VideoIO {
public:
    FileIO(std::string video_in);
    FileIO(int camera_idx);

    cv::Rect getInitRectangle() override;
    void outputBoundingBox(const cv::Rect & bbox) override;
    int getNextFileName(char * fName) override;    
    int getNextImage(cv::UMat &img) override;
    int getNextImage(cv::Mat & img) override;
    int getImageNum() const override;

private:
    cv::VideoCapture capture;
    std::string filename;
    int num = 0;
    std::ifstream rect_file;
    std::string rect_line;
};

#endif // VIDEOIO_HPP
