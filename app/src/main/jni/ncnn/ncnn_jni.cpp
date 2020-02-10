#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>

#include <jni.h>

#include <string>
#include <sstream>
#include <vector>



#include <sys/time.h>
#include <unistd.h>
#include <mat.h>
#include <net.h>

// ncnn
#include "include/net.h"
#include "include/allocator.h"
#include "include/mat.h"
#include "include/net.h"
#include "MobileFaceNet.h"
//opencv
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "detect.id.h"
#include "detect.mem.h"
#include "embed.id.h"
#include "embed.mem.h"


static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

struct Object
{
    //cv::Rect_<float> rect;
    float x, y, width, height;
    int label;
    float prob;
};

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "Ncnn", "%.2fms   %s", elasped, comment);
}

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

//static ncnn::Mat ncnn_param;
//static ncnn::Mat ncnn_bin;
//static std::vector<std::string> ncnn_label;
//static ncnn::Net ncnnnet;
//static ncnn::Net facenet;
//static Detector detector("detect.bin");
//static Embedder embedder("embed.bin");
static Detector detector;
static Embedder embeder;
static std::vector<float> my_vec;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

#define NCNNJNI_METHOD(METHOD_NAME) \
  Java_com_davidchiu_ncnncam_Ncnn_##METHOD_NAME  // NOLINT

extern "C" {

// public native boolean Init(byte[] param, byte[] bin, byte[] words);
//JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words, jobject assetManager)
JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jobject assetManager)
{
    detector.Init("detect.bin", env, assetManager);
    embeder.Init("embed.bin", env, assetManager);

    // Load image
    int width = 320;
    int height = 240;
    std::string face_img = "/storage/emulated/0/Dcim/Camera/IMG_20200130_152037.jpg";
    cv::Mat cv_img_mat = cv::imread(face_img.c_str(), cv::IMREAD_COLOR);
    cv::Mat cv_img_mat_rs;
    cv::Size new_size(width, height);
    cv::resize(cv_img_mat, cv_img_mat_rs, new_size);
    __android_log_print(ANDROID_LOG_DEBUG, "init cv", "width: %d, height: %d", cv_img_mat.cols, cv_img_mat.rows);
    __android_log_print(ANDROID_LOG_DEBUG, "init cv resize", "width: %d, height: %d", cv_img_mat_rs.cols, cv_img_mat_rs.rows);
    ncnn::Mat ncnn_img_mat = ncnn::Mat::from_pixels_resize(cv_img_mat_rs.data, ncnn::Mat::PIXEL_BGR, width, height, width, height);
    __android_log_print(ANDROID_LOG_DEBUG, "init ncnn size", "width: %d, height: %d", ncnn_img_mat.w, ncnn_img_mat.h);
    std::vector<bbox> boxes;
    int top_w, bottom_w, left_w, right_w;
    detector.Detect(ncnn_img_mat, boxes);
    ncnn::Mat face_crop;
    ncnn::Mat face_crop_rs;
    for (int i = 0; i < boxes.size(); ++i) {
        top_w = boxes[i].y1 * ncnn_img_mat.h;
        bottom_w = ncnn_img_mat.h - boxes[i].y2 * ncnn_img_mat.h;
        left_w = boxes[i].x1 * ncnn_img_mat.w;
        right_w = ncnn_img_mat.w - boxes[i].x2 * ncnn_img_mat.w;
        ncnn::copy_cut_border(ncnn_img_mat, face_crop, top_w, bottom_w, left_w, right_w);
        ncnn::resize_bilinear(face_crop, face_crop_rs, 112, 112);
//        std::vector<float> vec;
        my_vec = embeder.Embed(face_crop_rs);
    }
    return JNI_TRUE;
}


JNIEXPORT jarray JNICALL NCNNJNI_METHOD(nativeDetect)(JNIEnv* env, jobject thiz, jobject bitmap)
{
    bench_start();
    // ncnn from bitmap
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    int w = 320;
    int h = 240;
//    int w = 640;
//    int h = 480;
    __android_log_print(ANDROID_LOG_DEBUG, "yolov2ncnn", "image size: %dx%d", width, height);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    std::vector<bbox> boxes;
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, w, h);
    detector.Detect(in, boxes);
    int box_num = boxes.size() * 6 + 1;
    __android_log_print(ANDROID_LOG_DEBUG, "output size", "%d", box_num);
    jfloat output[box_num];
    output[0] = 6;
    ncnn::Mat face_crop;
    ncnn::Mat face_crop_rs;
    int top_w, bottom_w, left_w, right_w;
    float cosine_threshold = 0.6;
    for (int i = 0; i < boxes.size(); ++i) {
        // face embed
        top_w = boxes[i].y1 * in.h;
        bottom_w = in.h - boxes[i].y2 * in.h;
        left_w = boxes[i].x1 * in.w;
        right_w = in.w - boxes[i].x2 * in.w;
        ncnn::copy_cut_border(in, face_crop, top_w, bottom_w, left_w, right_w);
        ncnn::resize_bilinear(face_crop, face_crop_rs, 112, 112);
        std::vector<float> vec;
        vec = embeder.Embed(face_crop_rs);
        __android_log_print(ANDROID_LOG_DEBUG, "embed output", "%d", vec.size());
        float dist = embeder.cosine_distance(vec, my_vec);
        __android_log_print(ANDROID_LOG_DEBUG, "cosine distance", "%.2f", dist);
        if (dist < cosine_threshold) {
            output[i*6+1] = 1;
        }
        else {
            output[i*6+1] = 0;
        }

        output[i*6+2] = boxes[i].s;
        output[i*6+3] = boxes[i].x1;
        output[i*6+4] = boxes[i].y1;
        output[i*6+5] = boxes[i].x2;
        output[i*6+6] = boxes[i].y2;
    }

    jfloatArray jOutputData = env->NewFloatArray(boxes.size()*6+1);
    env->SetFloatArrayRegion(jOutputData, 0, boxes.size()*6+1, output);  // copy
    return jOutputData;
  }
}
