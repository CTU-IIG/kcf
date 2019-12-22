//#include "matutil.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//
//cv::Mat MatUtil::plane(uint scale, uint feature, cv::Mat &host) {
//    assert(host.dims == 4);
//    assert(int(scale) < host.size[0]);
//    assert(int(feature) < host.size[1]);
//    return cv::Mat(host.size[2], host.size[3], host.type(), host.ptr(scale, feature));
//}
//
///*
// * Function for getting cv::Mat header referencing features, height and width of the input matrix.
// * Presumes input matrix of 4 dimensions with format: {scales, features, height, width}
// **/
//cv::Mat MatUtil::scale(uint scale, cv::Mat &host) {
//    assert(host.dims == 4);
//    assert(int(scale) < host.size[0]);
//    return cv::Mat(3, std::vector<int>({host.size[1], host.size[2], host.size[3]}).data(), host.type(), host.ptr(scale));
//}
//   
//
///*
// * Sets the source as channel number idx of target matrix.
// **/
//void MatUtil::set_channel(int idx, cv::Mat &source, cv::Mat &target)
//{
//    assert(idx < target.channels());
//    assert(source.channels() == 1);
//    int from_to[] = { 0,idx };
//    cv::mixChannels( &source, 1, &target, 1, from_to, 1 );
//}