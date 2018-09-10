/*
    - c++ wrapper for the piotr toolbox
    Created by Tomas Vojir, 2014
*/


#ifndef FHOG_HEADER_7813784354687
#define FHOG_HEADER_7813784354687

#include <vector>
#include <opencv2/opencv.hpp>

#include "gradientMex.h"
#include "threadctx.hpp"
#include "pragmas.h"

class FHoG
{
public:
    //description: extract hist. of gradients(use_hog == 0), hog(use_hog == 1) or fhog(use_hog == 2)
    //input: float one channel image as input, hog type
    //return: computed descriptor
    static void extract(const cv::Mat & img, ThreadCtx & vars,int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
    {
        // d image dimension -> gray image d = 1
        // h, w -> height, width of image
        // full -> ??
        // I -> input image, M, O -> mag, orientation OUTPUT
        int h = img.rows, w = img.cols, d = 1;
        bool full = true;
        if (h < 2 || w < 2) {
            std::cerr << "I must be at least 2x2." << std::endl;
            return;
        }

//        //image rows-by-rows
//        float * I = new float[h*w];
//        for (int y = 0; y < h; ++y) {
//            const float * row_ptr = img.ptr<float>(y);
//            for (int x = 0; x < w; ++x) {
//                I[y*w + x] = row_ptr[x];
//            }
//        }

        //image cols-by-cols
        float * I = new float[h*w];
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                I[x*h + y] = img.at<float>(y, x)/255.f;
            }
        }

        float *M = new float[h*w], *O = new float[h*w];
        gradMag(I, M, O, h, w, d, full);

        int n_chns = (use_hog == 0) ? n_orients : (use_hog==1 ? n_orients*4 : n_orients*3+5);
        int hb = h/bin_size, wb = w/bin_size;

        float *H = new float[hb*wb*n_chns];
        memset(H, 0, hb*wb*n_chns*sizeof(float));

        if (use_hog == 0) {
            full = false;   //by default
            gradHist( M, O, H, h, w, bin_size, n_orients, soft_bin, full );
        } else if (use_hog == 1) {
            full = false;   //by default
            hog( M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip );
        } else {
            fhog( M, O, H, h, w, bin_size, n_orients, soft_bin, clip );
        }

        //convert, assuming row-by-row-by-channel storage
        int n_res_channels = (use_hog == 2) ? n_chns-1 : n_chns;    //last channel all zeros for fhog
        for (int i = 0; i < n_res_channels; ++i) {
            //output rows-by-rows
//            cv::Mat desc(hb, wb, CV_32F, (H+hb*wb*i));

            //output cols-by-cols
            cv::Mat desc(hb, wb, CV_32F);
            for (int x = 0; x < wb; ++x) {
                for (int y = 0; y < hb; ++y) {
                    desc.at<float>(y,x) = H[i*hb*wb + x*hb + y];
                }
            }
            BIG_BATCH_OMP_ORDERED
            vars.patch_feats.push_back(desc.clone());
        }

        //clean
        delete [] I;
        delete [] M;
        delete [] O;
        delete [] H;

        return;
    }

};

#endif //FHOG_HEADER_7813784354687
