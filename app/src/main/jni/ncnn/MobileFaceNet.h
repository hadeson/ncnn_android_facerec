//
// Created by sonct on 10/02/2020.
//

#ifndef NCNNCAM_MOBILEFACENET_H
#define NCNNCAM_MOBILEFACENET_H

#include <algorithm>
#include <android/asset_manager_jni.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "include/net.h"
#include <math.h>
#include <vector>
#include <stdexcept>
#include <jni.h>


struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

class Detector
{

public:
    Detector();

    void Init(const std::string &model_bin, JNIEnv* env, jobject assetManager);

    Detector(const std::string &model_bin, JNIEnv* env, jobject assetManager);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, int img_w, int img_h, float NMS_THRESH);

//    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);
    void Detect(ncnn::Mat& in, std::vector<bbox>& boxes);

    void face_align(cv::Mat& face, bbox face_box);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    ~Detector();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;

    ncnn::Net *Net;
};

class Embedder
{

public:
    Embedder();

    void Init(const std::string &model_bin, JNIEnv* env, jobject assetManager);

    Embedder(const std::string &model_bin, JNIEnv* env, jobject assetManager);

    inline void Release();

    std::vector<float> Embed(ncnn::Mat& in, bool normalize);

    inline void SetDefaultParams();

    float get_l2_similarity(std::vector<float> v1, std::vector<float> v2);

    float cosine_distance(std::vector<float> A, std::vector<float>B);

    void normalize(std::vector<float>& arr);

    ~Embedder();

public:
    // float _nms;
    // float _threshold;
    float _mean_val[3];
    // bool _retinaface;

    ncnn::Net *Net;
};
#endif //NCNNCAM_MOBILEFACENET_H
