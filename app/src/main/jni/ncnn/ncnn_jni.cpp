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

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

//static Detector detector("detect.bin");
//static Embedder embedder("embed.bin");
static Detector detector;
static Embedder embeder;
//static std::vector<float> my_vec;
//static std::vector<std::vector<float>> my_vecs;
static std::vector<std::vector<std::vector<float>>> my_vecs;
static int vec_size = 128;
std::string align_face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_123859.jpg";
//std::string resize_face_img = "/storage/emulated/0/Dcim/Camera/SAVE_20200212_123851.jpg";
std::string resize_full_img = "/storage/emulated/0/Download/full.png";
std::string resize_face_img = "/storage/emulated/0/Download/face.png";
std::string face_img_ori = "/storage/emulated/0/Download/face_ori.png";


#define NCNNJNI_METHOD(METHOD_NAME) \
  Java_com_davidchiu_ncnncam_Ncnn_##METHOD_NAME  // NOLINT

extern "C" {

JNIEXPORT jarray JNICALL  NCNNJNI_METHOD(embed)(JNIEnv* env, jobject thiz, jobject assetManager, jstring img_path) {
    if (!detector._loaded) {
        detector.Init("detect.bin", env, assetManager);
        embeder.Init("embed.bin", env, assetManager);
    }
    const char *native_img_path = env->GetStringUTFChars(img_path, 0);
//    __android_log_print(ANDROID_LOG_DEBUG, "Native img path", "%s", native_img_path);

    int width = 640;
    int height = 480;
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED img path", "%s", native_img_path);
    cv::Mat cv_img_mat_raw = cv::imread(native_img_path, cv::IMREAD_COLOR);
    env->ReleaseStringUTFChars(img_path, native_img_path);
    cv::Mat cv_img_mat;
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
    } else {
        cv_img_mat = cv_img_mat_raw.clone();
    }
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "width: %d, height: %d", cv_img_mat.cols, cv_img_mat.rows);
    ori_width = cv_img_mat.cols;
    ori_height = cv_img_mat.rows;
    cv::Mat cv_img_mat_rs;
    cv::Size new_size(width, height);
    cv::resize(cv_img_mat, cv_img_mat_rs, new_size, cv::INTER_AREA);
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv resize", "width: %d, height: %d", cv_img_mat_rs.cols, cv_img_mat_rs.rows);
    cv::imwrite(resize_full_img, cv_img_mat_rs);
    ncnn::Mat ncnn_img_mat = ncnn::Mat::from_pixels_resize(cv_img_mat_rs.data, ncnn::Mat::PIXEL_BGR,
                                                           width, height, width, height);
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED ncnn size", "width: %d, height: %d", ncnn_img_mat.w, ncnn_img_mat.h);

    std::vector<bbox> boxes;
    detector.Detect(ncnn_img_mat, boxes);
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "after detect, boxes: %d", boxes.size());

    if (boxes.size() == 0) {
        jfloatArray jOutputData = env->NewFloatArray(0);
        return jOutputData;
    }
    ncnn::Mat ncnn_face;
    cv::Mat cv_face;
    cv::Mat cv_face_rs;
    cv::Size face_size(112, 112);
    std::vector<std::vector<float>> new_vecs;
    int max_box_area = 0;
    int max_box_id = -1;
    for (int i = 0; i < boxes.size(); ++i) {
        int cv_x1 = boxes[i].x1 * ori_width;
        int cv_y1 = boxes[i].y1 * ori_height;
        int cv_x2 = boxes[i].x2 * ori_width;
        int cv_y2 = boxes[i].y2 * ori_height;
        int face_width = cv_x2 - cv_x1;
        int face_height = cv_y2 - cv_y1;
        int box_area = face_width * face_height;
        if (box_area > max_box_area) {
            max_box_area = box_area;
            max_box_id = i;
        }
    }
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "after get max size face");

    int cv_x1 = boxes[max_box_id].x1 * ori_width;
    int cv_y1 = boxes[max_box_id].y1 * ori_height;
    int cv_x2 = boxes[max_box_id].x2 * ori_width;
    int cv_y2 = boxes[max_box_id].y2 * ori_height;
    int face_width = cv_x2 - cv_x1;
    int face_height = cv_y2 - cv_y1;
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "before adjust landmark");
    for (int j = 0; j < 5; j++) {
        boxes[max_box_id].point[j]._x = (boxes[max_box_id].point[j]._x * ori_width - cv_x1) / face_width * 112;
        boxes[max_box_id].point[j]._y = (boxes[max_box_id].point[j]._y * ori_height - cv_y1) / face_height * 112;
    }
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "before rect");
    cv_face = cv_img_mat(cv::Rect(cv_x1, cv_y1, face_width, face_height)).clone();
    cv::imwrite(face_img_ori, cv_face);
    cv::resize(cv_face, cv_face_rs, face_size, cv::INTER_AREA);
    detector.face_align(cv_face_rs, boxes[max_box_id]);
    cv::imwrite(resize_face_img, cv_face_rs);
    ncnn_face = ncnn::Mat::from_pixels_resize(cv_face_rs.data, ncnn::Mat::PIXEL_RGB, 112, 112,
                                              112, 112);
//    __android_log_print(ANDROID_LOG_DEBUG, "face size ncnn", "width: %d, height: %d",
//                        ncnn_face.w, ncnn_face.h);
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "before embed");
    new_vecs.push_back(embeder.Embed(ncnn_face, false));

    // parse result to java
    int jOutputSize = new_vecs.size()*vec_size;
    jfloat output[jOutputSize];
    for (int i = 0; i < new_vecs.size(); i++) {
        for (int j = 0; j < vec_size; j++) {
            output[i*vec_size + j] = new_vecs[i][j];
        }
    }
    jfloatArray jOutputData = env->NewFloatArray(jOutputSize);
    env->SetFloatArrayRegion(jOutputData, 0, jOutputSize, output);  // copy
    __android_log_print(ANDROID_LOG_DEBUG, "EMBED cv", "finished");
    return jOutputData;
}

JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(load)(JNIEnv* env, jobject thiz, jobject assetManager, jintArray ids, jfloatArray features) {
    if (!detector._loaded) {
        detector.Init("detect.bin", env, assetManager);
        embeder.Init("embed.bin", env, assetManager);
    }

    jsize size = env->GetArrayLength( ids );
    std::vector<int> ids_vec( size );
    env->GetIntArrayRegion( ids, 0, size, &ids_vec[0] );

    size = env->GetArrayLength( features );
    std::vector<float> features_vec( size );
    env->GetFloatArrayRegion( features, 0, size, &features_vec[0] );

    __android_log_print(ANDROID_LOG_DEBUG, "LOAD", "id: %d ids[0]: %d, features: %d, my_vec: %d random_point: %.2f", ids_vec.size(), ids_vec[0], features_vec.size(), my_vecs[0].size(), features[100]);
    int cur_id = -1;
    std::vector<std::vector<float>> acc_features;
    for (int i = 0; i < ids_vec.size(); i++) {
        if (cur_id != ids_vec[i]) {
            if (cur_id != -1) {
                my_vecs.push_back(acc_features);
            }
            acc_features.clear();
            cur_id = ids_vec[i];
        }
        std::vector<float> feat;
        for (int j = i*vec_size; j < (i+1)*vec_size; j++) {
            feat.push_back(features_vec[j]);
        }
        acc_features.push_back(feat);
    }
    my_vecs.push_back(acc_features);
    __android_log_print(ANDROID_LOG_DEBUG, "LOAD", "id: %d features: %d, my_vec: %d random_point: %.2f", ids_vec.size(), features_vec.size(), my_vecs[0].size(), features[100]);

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL  NCNNJNI_METHOD(init)(JNIEnv* env, jobject thiz, jobject assetManager, jintArray ids, jfloatArray features)
{
    if (!detector._loaded) {
        detector.Init("detect.bin", env, assetManager);
        embeder.Init("embed.bin", env, assetManager);
    }
    if (my_vecs.size() > 0) {
        return JNI_TRUE;
    }
    jsize size = env->GetArrayLength( ids );
    std::vector<int> ids_vec( size );
    env->GetIntArrayRegion( ids, 0, size, &ids_vec[0] );

    size = env->GetArrayLength( features );
    std::vector<float> features_vec( size );
    env->GetFloatArrayRegion( features, 0, size, &features_vec[0] );

//    __android_log_print(ANDROID_LOG_DEBUG, "INIT", "id: %d features: %d, my_vec: %d random_point: %.2f", ids_vec.size(), features_vec.size(), my_vecs[0].size(), features[100]);
    int cur_id = -1;
    std::vector<std::vector<float>> acc_features;
    for (int i = 0; i < ids_vec.size(); i++) {
        if (cur_id != ids_vec[i]) {
            if (cur_id != -1) {
                my_vecs.push_back(acc_features);
            }
            acc_features.clear();
            cur_id = ids_vec[i];
        }
        std::vector<float> feat;
        for (int j = i*vec_size; j < (i+1)*vec_size; j++) {
            feat.push_back(features_vec[j]);
        }
        acc_features.push_back(feat);
    }
    my_vecs.push_back(acc_features);
    __android_log_print(ANDROID_LOG_DEBUG, "INIT", "id: %d features: %d, my_vec: %d random_point: %.2f", ids_vec.size(), features_vec.size(), my_vecs[0].size(), features[100]);

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
//        cv::imwrite(resize_face_img, cv_face_rs);

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
//        output[i*6+1] = 0;
        float min_dist = 1000000;
        int min_id = -1;
        __android_log_print(ANDROID_LOG_DEBUG, "MY_VEC", "%d", my_vecs.size());
        for (int j = 0; j < my_vecs.size(); j++) {
            for (int k = 0; k < my_vecs[j].size(); k++) {
                float dist = embeder.cosine_distance(vec, my_vecs[j][k]);
                __android_log_print(ANDROID_LOG_DEBUG, "cosine distance", "%.2f", dist);
                if ((dist < cosine_threshold) && (dist < min_dist)) {
                    min_dist = dist;
                    min_id = j;
                }
            }
        }

        __android_log_print(ANDROID_LOG_DEBUG, "min_id", "%d", min_id);
        output[i*6+1] = min_id;
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
