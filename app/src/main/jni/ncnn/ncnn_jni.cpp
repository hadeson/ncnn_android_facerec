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
std::string align_face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_123859.jpg";
std::string resize_face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_123851.jpg";


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
//    int width = 320;
//    int height = 240;
    int width = 640;
    int height = 480;
//    int width_rv = 480;
//    int height_rv = 640;
    if (my_vecs.size() > 0) {
        return JNI_TRUE;
    }
    for (int ii = 0; ii < face_imgs.size(); ii++) {
        cv::Mat cv_img_mat_raw = cv::imread(face_imgs[ii].c_str(), cv::IMREAD_COLOR);
        cv::Mat cv_img_mat;
//        cv::Mat cv_img_mat_raw = cv_img_mat.clone();
        // resize image to retain info
        // center crop in case of h > w
        int ori_width = cv_img_mat_raw.cols;
        int ori_height = cv_img_mat_raw.rows;
        if (ori_height > ori_width) {
            int x1 = 0;
            int y1 = 0.2 * cv_img_mat_raw.rows;
            int new_width = cv_img_mat_raw.cols;
            int new_height = cv_img_mat_raw.rows * 0.65;
            cv_img_mat = cv_img_mat_raw(cv::Rect(x1, y1, new_width, new_height)).clone();
        }
        else {
            cv_img_mat = cv_img_mat_raw.clone();
        }
        ori_width = cv_img_mat.cols;
        ori_height = cv_img_mat.rows;
        cv::Mat cv_img_mat_rs;
        cv::Size new_size(width, height);
        cv::resize(cv_img_mat, cv_img_mat_rs, new_size);
//        __android_log_print(ANDROID_LOG_DEBUG, "init cv", "width: %d, height: %d", cv_img_mat.cols, cv_img_mat.rows);
//        __android_log_print(ANDROID_LOG_DEBUG, "init cv resize", "width: %d, height: %d", cv_img_mat_rs.cols, cv_img_mat_rs.rows);
        ncnn::Mat ncnn_img_mat = ncnn::Mat::from_pixels_resize(cv_img_mat_rs.data, ncnn::Mat::PIXEL_BGR, width, height, width, height);
//        __android_log_print(ANDROID_LOG_DEBUG, "init ncnn size", "width: %d, height: %d", ncnn_img_mat.w, ncnn_img_mat.h);

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
            for (int j = 0; j < 5; j++) {
                boxes[i].point[j]._x = (boxes[i].point[j]._x * ori_width - cv_x1) / face_width * 112;
                boxes[i].point[j]._y = (boxes[i].point[j]._y * ori_height - cv_y1) / face_height * 112;
            }
            cv_face = cv_img_mat(cv::Rect(cv_x1, cv_y1, face_width, face_height)).clone();
            cv::resize(cv_face, cv_face_rs, face_size);
            detector.face_align(cv_face_rs, boxes[i]);
//            cv::cvtColor(cv_face_rs, cv_face_rs, cv::COLOR_BGR2RGB);
//            __android_log_print(ANDROID_LOG_DEBUG, "face size", "width: %d, height: %d", face_width, face_height);
//        ncnn_face = ncnn::Mat::from_pixels_resize(cv_face.data, ncnn::Mat::PIXEL_BGR, face_width, face_height, 112, 112);
//        ncnn_face = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_BGR, 112, 112, 112, 112);
            ncnn_face = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_RGB, 112, 112, 112, 112);
            __android_log_print(ANDROID_LOG_DEBUG, "face size ncnn", "width: %d, height: %d", ncnn_face.w, ncnn_face.h);
            my_vecs.push_back(embeder.Embed(ncnn_face, false));
        }
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
//    int w = 320;
//    int h = 240;
    int w = 640;
    int h = 480;
    cv::Size img_size(w, h);

    void* pixels = 0;
    int ret = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    cv::Mat tmp_mat(info.height, info.width, CV_8UC4, pixels);
    cv::Mat cv_img_mat;
    cv::Mat cv_img_mat_raw;
    cv::Mat cv_img_mat_rs;
    cv::cvtColor(tmp_mat, cv_img_mat_raw, cv::COLOR_RGBA2BGR);
    cv_img_mat = cv_img_mat_raw(cv::Rect(0, 0, 480, 360));
//    cv::imwrite(align_face_img, cv_img_mat); // write non resize
//    cv::cvtColor(tmp_mat, cv_img_mat, cv::COLOR_RGBA2RGB);
    cv::resize(cv_img_mat, cv_img_mat_rs, img_size);
//    cv::imwrite(resize_face_img, cv_img_mat_rs); //write resize
//    cv::Mat cv_img_mat_raw = cv_img_mat.clone();
    int ori_width = cv_img_mat.cols;
    int ori_height = cv_img_mat.rows;
    AndroidBitmap_unlockPixels(env, bitmap);

    __android_log_print(ANDROID_LOG_DEBUG, "yolov2ncnn", "image size: %dx%d", width, height);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    std::vector<bbox> boxes;
//    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, w, h);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(cv_img_mat_rs.data, ncnn::Mat::PIXEL_BGR, w, h, w, h);
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
    float cosine_threshold = 0.5;
    for (int i = 0; i < boxes.size(); ++i) {
        int cv_x1 = boxes[i].x1 * ori_width;
        int cv_y1 = boxes[i].y1 * ori_height;
        int cv_x2 = boxes[i].x2 * ori_width;
        int cv_y2 = boxes[i].y2 * ori_height;
//        int cv_x1 = boxes[i].x1 * w;
//        int cv_y1 = boxes[i].y1 * h;
//        int cv_x2 = boxes[i].x2 * w;
//        int cv_y2 = boxes[i].y2 * h;
        int face_width = cv_x2-cv_x1;
        int face_height = cv_y2-cv_y1;
//        cv_y1 += face_height * 0.05;
//        face_height -= face_height * 0.15;
        for (int j = 0; j < 5; j++) {
            boxes[i].point[j]._x = (boxes[i].point[j]._x * ori_width - cv_x1) / face_width * 112;
            boxes[i].point[j]._y = (boxes[i].point[j]._y * ori_height - cv_y1) / face_height * 112;
//            boxes[i].point[j]._x = (boxes[i].point[j]._x * ori_width - cv_x1);
//            boxes[i].point[j]._y = (boxes[i].point[j]._y * ori_height - cv_y1);
//            boxes[i].point[j]._x = boxes[i].point[j]._x * ori_width;
//            boxes[i].point[j]._y = boxes[i].point[j]._y * ori_height;
//            boxes[i].point[j]._x = boxes[i].point[j]._x * w;
//            boxes[i].point[j]._y = boxes[i].point[j]._y * h;
        }
//        cv_face = cv_img_mat(cv::Rect(cv_x1, cv_y1, face_width, face_height));
        cv_face = cv_img_mat_raw(cv::Rect(cv_x1, cv_y1, face_width, face_height)).clone();
//        cv::cvtColor(cv_face, cv_face, cv::COLOR_RGB2BGR);
        cv::resize(cv_face, cv_face_rs, face_size);
        __android_log_print(ANDROID_LOG_DEBUG, "landmarks", "p1: %.4f %.4f, p2: %.4f %.4f, p3: %.4f %.4f, p4: %.4f %.4f, p5: %.4f %.4f", boxes[i].point[0]._x*112, boxes[i].point[0]._y*112, boxes[i].point[1]._x*112, boxes[i].point[1]._y*112, boxes[i].point[2]._x*112, boxes[i].point[2]._y*112, boxes[i].point[3]._x*112, boxes[i].point[3]._y*112, boxes[i].point[4]._x*112, boxes[i].point[4]._y*112);
        cv::circle(cv_face_rs, cv::Point(boxes[i].point[0]._x, boxes[i].point[0]._y), 1, cv::Scalar(0, 0, 225), 4);
        cv::circle(cv_face_rs, cv::Point(boxes[i].point[1]._x, boxes[i].point[1]._y), 1, cv::Scalar(0, 255, 225), 4);
        cv::circle(cv_face_rs, cv::Point(boxes[i].point[2]._x, boxes[i].point[2]._y), 1, cv::Scalar(255, 0, 225), 4);
        cv::circle(cv_face_rs, cv::Point(boxes[i].point[3]._x, boxes[i].point[3]._y), 1, cv::Scalar(0, 255, 0), 4);
        cv::circle(cv_face_rs, cv::Point(boxes[i].point[4]._x, boxes[i].point[4]._y), 1, cv::Scalar(255, 0, 0), 4);
        cv::imwrite(resize_face_img, cv_face_rs);

//        cv::cvtColor(cv_img_mat, cv_img_mat, cv::COLOR_RGB2BGR);
//        cv::circle(cv_img_mat, cv::Point(boxes[i].point[0]._x, boxes[i].point[0]._y), 1, cv::Scalar(0, 0, 225), 4);
//        cv::circle(cv_img_mat, cv::Point(boxes[i].point[1]._x, boxes[i].point[1]._y), 1, cv::Scalar(0, 255, 225), 4);
//        cv::circle(cv_img_mat, cv::Point(boxes[i].point[2]._x, boxes[i].point[2]._y), 1, cv::Scalar(255, 0, 225), 4);
//        cv::circle(cv_img_mat, cv::Point(boxes[i].point[3]._x, boxes[i].point[3]._y), 1, cv::Scalar(0, 255, 0), 4);
//        cv::circle(cv_img_mat, cv::Point(boxes[i].point[4]._x, boxes[i].point[4]._y), 1, cv::Scalar(255, 0, 0), 4);
//        cv::imwrite(resize_face_img, cv_img_mat);

        detector.face_align(cv_face_rs, boxes[i]);
        cv::imwrite(align_face_img, cv_face_rs);
//        cv::imwrite(align_face_img, cv_face_rs);
        __android_log_print(ANDROID_LOG_DEBUG, "face size", "width: %d, height: %d", face_width, face_height);
//        face_crop_rs = ncnn::Mat::from_pixels_resize(cv_face.data, ncnn::Mat::PIXEL_BGR, face_width, face_height, 112, 112);
//        face_crop_rs = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_BGR, 112, 112, 112, 112);
        cv::cvtColor(cv_face_rs, cv_face_rs, cv::COLOR_BGR2RGB);
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
        output[i*6+4] = boxes[i].y1 / height * 360;
        output[i*6+5] = boxes[i].x2;
        output[i*6+6] = boxes[i].y2 / height * 360;
    }

    jfloatArray jOutputData = env->NewFloatArray(boxes.size()*6+1);
    env->SetFloatArrayRegion(jOutputData, 0, boxes.size()*6+1, output);  // copy
    return jOutputData;
  }
}
