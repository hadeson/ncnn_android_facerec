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

//#include "detect.id.h"
//#include "detect.mem.h"
//#include "embed.id.h"
//#include "embed.mem.h"


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
//static std::vector<float> my_vec;
static std::vector<std::vector<float>> my_vecs;
static std::vector<std::string> face_imgs = {
    "/storage/emulated/0/Dcim/Camera/IMG_20200130_152037.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171702.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171659.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171656.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171654.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171652.jpg",
    "/storage/emulated/0/Dcim/Camera/IMG_20200212_171648.jpg",
};


#define NCNNJNI_METHOD(METHOD_NAME) \
  Java_com_davidchiu_ncnncam_Ncnn_##METHOD_NAME  // NOLINT

extern "C" {

// public native boolean Init(byte[] param, byte[] bin, byte[] words);
//JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jbyteArray param, jbyteArray bin, jbyteArray words, jobject assetManager)
JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jobject assetManager)
{
    detector.Init("detect.bin", env, assetManager);
    embeder.Init("embed.bin", env, assetManager);
//    embeder.Init("mobilefacenet_v2.bin", env, assetManager);
//    embeder.Init("arcface.bin", env, assetManager);

    // Load image
    int width = 320;
    int height = 240;
    if (my_vecs.size() > 0) {
        return JNI_TRUE;
    }
    for (int ii = 0; ii < face_imgs.size(); ii++) {
        cv::Mat cv_img_mat = cv::imread(face_imgs[ii].c_str(), cv::IMREAD_COLOR);
        cv::Mat cv_img_mat_raw = cv_img_mat.clone();
        int ori_width = cv_img_mat.cols;
        int ori_height = cv_img_mat.rows;
        cv::Mat cv_img_mat_rs;
        cv::Size new_size(width, height);
        cv::resize(cv_img_mat, cv_img_mat_rs, new_size);
        __android_log_print(ANDROID_LOG_DEBUG, "init cv", "width: %d, height: %d", cv_img_mat.cols, cv_img_mat.rows);
        __android_log_print(ANDROID_LOG_DEBUG, "init cv resize", "width: %d, height: %d", cv_img_mat_rs.cols, cv_img_mat_rs.rows);
        ncnn::Mat ncnn_img_mat = ncnn::Mat::from_pixels_resize(cv_img_mat_rs.data, ncnn::Mat::PIXEL_BGR, width, height, width, height);
        __android_log_print(ANDROID_LOG_DEBUG, "init ncnn size", "width: %d, height: %d", ncnn_img_mat.w, ncnn_img_mat.h);

        std::vector<bbox> boxes;
        detector.Detect(ncnn_img_mat, boxes);

        ncnn::Mat ncnn_face;
        cv::Mat cv_face;
        cv::Mat cv_face_rs;
        cv::Size face_size(112, 112);
        for (int i = 0; i < boxes.size(); ++i) {
            int cv_x1 = boxes[i].x1 * ori_width;
            int cv_y1 = boxes[i].y1 * ori_height;
            int cv_x2 = boxes[i].x2 * ori_width;
            int cv_y2 = boxes[i].y2 * ori_height;
            int face_width = cv_x2-cv_x1;
            int face_height = cv_y2-cv_y1;
            cv_y1 += face_height * 0.05;
            face_height -= face_height * 0.15;
//        cv_face = cv_img_mat(cv::Rect(cv_x1, cv_y1, face_width, face_height));
            cv_face = cv_img_mat_raw(cv::Rect(cv_x1, cv_y1, face_width, face_height));
            cv::cvtColor(cv_face, cv_face, cv::COLOR_BGR2RGB);
            detector.face_align(cv_face, boxes[i]);
            cv::resize(cv_face, cv_face_rs, face_size);
            __android_log_print(ANDROID_LOG_DEBUG, "face size", "width: %d, height: %d", face_width, face_height);
//        ncnn_face = ncnn::Mat::from_pixels_resize(cv_face.data, ncnn::Mat::PIXEL_BGR, face_width, face_height, 112, 112);
//        ncnn_face = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_BGR, 112, 112, 112, 112);
            ncnn_face = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_RGB, 112, 112, 112, 112);
            __android_log_print(ANDROID_LOG_DEBUG, "face size ncnn", "width: %d, height: %d", ncnn_face.w, ncnn_face.h);
            my_vecs.push_back(embeder.Embed(ncnn_face, false));
        }
    }
//    std::string face_img = "/storage/emulated/0/Dcim/Camera/IMG_20200130_152037.jpg";
//    // Chipu
//    std::string face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_115217.jpg";
//    // Trong
//    std::string face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_115217.jpg";
//    // Trump
//    std::string face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_123859.jpg";

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

    void* pixels = 0;
    int ret = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    cv::Mat tmp_mat(info.height, info.width, CV_8UC4, pixels);
    cv::Mat cv_img_mat;
//    cv::cvtColor(tmp_mat, cv_img_mat, cv::COLOR_RGBA2BGR);
    cv::cvtColor(tmp_mat, cv_img_mat, cv::COLOR_RGBA2RGB);
    cv::Mat cv_img_mat_raw = cv_img_mat.clone();
    int ori_width = cv_img_mat.cols;
    int ori_height = cv_img_mat.rows;
    AndroidBitmap_unlockPixels(env, bitmap);

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
    cv::Mat cv_face;
    cv::Mat cv_face_rs;
    cv::Size face_size(112, 112);
    int top_w, bottom_w, left_w, right_w;
    float cosine_threshold = 0.5;
    for (int i = 0; i < boxes.size(); ++i) {
        // face embed
//        top_w = boxes[i].y1 * in.h;
//        bottom_w = in.h - boxes[i].y2 * in.h;
//        left_w = boxes[i].x1 * in.w;
//        right_w = in.w - boxes[i].x2 * in.w;
//        ncnn::copy_cut_border(in, face_crop, top_w, bottom_w, left_w, right_w);
//        ncnn::resize_bilinear(face_crop, face_crop_rs, 112, 112);

        int cv_x1 = boxes[i].x1 * ori_width;
        int cv_y1 = boxes[i].y1 * ori_height;
        int cv_x2 = boxes[i].x2 * ori_width;
        int cv_y2 = boxes[i].y2 * ori_height;
        int face_width = cv_x2-cv_x1;
        int face_height = cv_y2-cv_y1;
        cv_y1 += face_height * 0.05;
        face_height -= face_height * 0.15;
//        cv_face = cv_img_mat(cv::Rect(cv_x1, cv_y1, face_width, face_height));
        cv_face = cv_img_mat_raw(cv::Rect(cv_x1, cv_y1, face_width, face_height));
        detector.face_align(cv_face, boxes[i]);
        cv::resize(cv_face, cv_face_rs, face_size);
        __android_log_print(ANDROID_LOG_DEBUG, "face size", "width: %d, height: %d", face_width, face_height);
//        face_crop_rs = ncnn::Mat::from_pixels_resize(cv_face.data, ncnn::Mat::PIXEL_BGR, face_width, face_height, 112, 112);
//        face_crop_rs = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_BGR, 112, 112, 112, 112);
        face_crop_rs = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_RGB, 112, 112, 112, 112);

        std::vector<float> vec = embeder.Embed(face_crop_rs, false);
        __android_log_print(ANDROID_LOG_DEBUG, "embed output", "%d", vec.size());
        output[i*6+1] = 0;
        for (int j = 0; j < my_vecs.size(); j++) {
            float dist = embeder.cosine_distance(vec, my_vecs[j]);
            __android_log_print(ANDROID_LOG_DEBUG, "cosine distance", "%.2f", dist);
            if (dist < cosine_threshold) {
                output[i*6+1] = 1;
            }
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
