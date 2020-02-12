//
// Created by sonct on 10/02/2020.
//
#include "MobileFaceNet.h"
#include "detect.id.h"
#include "detect.mem.h"
#include "embed.id.h"
#include "embed.mem.h"
//#include "mobilefacenet_v2.id.h"
//#include "mobilefacenet_v2.mem.h"
//#include "arcface.id.h"
//#include "arcface.mem.h"


Detector::Detector():
        _nms(0.4),
        _threshold(0.6),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(false),
        Net(new ncnn::Net())
{
}

inline void Detector::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Detector::Detector(const std::string &model_bin, JNIEnv* env, jobject assetManager):
        _nms(0.7),
        _threshold(0.6),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
    Init(model_bin, env, assetManager);
}

void Detector::Init(const std::string &model_bin, JNIEnv* env, jobject assetManager)
{
    static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    static ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    Net->opt = opt;
    int ret = Net->load_param(detect_param_bin);
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    ret = Net->load_model(mgr, model_bin.c_str());
}

void Detector::Detect(ncnn::Mat& in, std::vector<bbox>& boxes)
{
//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
//            bgr.data, ncnn::Mat::PIXEL_BGR,
//            bgr.cols, bgr.rows, bgr.cols, bgr.rows
//    );

    in.substract_mean_normalize(_mean_val, 0);
    ncnn::Extractor ex = Net->create_extractor();
//    ex.set_light_mode(true);
//    ex.set_num_threads(4);
    ex.input(detect_param_id::BLOB_input0, in);
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract(detect_param_id::BLOB_output0, out);

    // class
    ex.extract(detect_param_id::BLOB_530, out1);

    //landmark
    ex.extract(detect_param_id::BLOB_529, out2);


    std::vector<box> anchor;

    create_anchor(anchor, in.w, in.h);

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);

    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2);
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2);
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2);
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2);
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx );
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy );
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    for (int j = 0; j < total_box.size(); ++j)
    {
        boxes.push_back(total_box[j]);
    }
}

void Detector::face_align(cv::Mat& face, bbox face_box) {
    // construct source matrix
    cv::Mat src(5, 2, CV_64FC1);
    src.at<double>(0, 0) = 38.2946;
    src.at<double>(0, 1) = 51.6963;
    src.at<double>(1, 0) = 73.5318;
    src.at<double>(1, 1) = 51.5014;
    src.at<double>(2, 0) = 56.0252;
    src.at<double>(2, 1) = 71.7366;
    src.at<double>(3, 0) = 41.5493;
    src.at<double>(3, 1) = 92.3655;
    src.at<double>(4, 0) = 70.7299;
    src.at<double>(4, 1) = 92.2041;

    cv::Mat dst(5, 2, CV_64FC1);
    for (int i = 0; i < 5; i++) {
        dst.at<double>(i, 0) = face_box.point[i]._x;
        dst.at<double>(i, 1) = face_box.point[i]._y;
    }
    cv::Mat affine_mat = cv::getAffineTransform(src, dst);
    cv::Mat warped_face = cv::Mat::zeros(face.rows, face.cols, face.type());
    cv::warpAffine(face, warped_face, affine_mat, face.size(), cv::INTER_LINEAR);
    face = warped_face.clone();
}

inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detector::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;

}

Detector::~Detector(){
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

Embedder::Embedder():
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
}

inline void Embedder::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Embedder::Embedder(const std::string &model_bin, JNIEnv* env, jobject assetManager):
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
    Init(model_bin, env, assetManager);
}

void Embedder::Init(const std::string &model_bin, JNIEnv* env, jobject assetManager)
{
    static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    static ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    Net->opt = opt;
//    int ret = Net->load_param(arcface_param_bin);
    int ret = Net->load_param(embed_param_bin);
//    int ret = Net->load_param(mobilefacenet_v2_param_bin);
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    ret = Net->load_model(mgr, model_bin.c_str());
}

Embedder::~Embedder() {
    Release();
}


std::vector<float> Embedder::Embed(ncnn::Mat& in, bool normalize)
{
    if (normalize) {
        in.substract_mean_normalize(_mean_val, 0);
    }
//    char out_w[80];
    ncnn::Extractor ex = Net->create_extractor();
//    ex.set_light_mode(true);
//    ex.set_num_threads(4);
//    ex.input(arcface_param_id::BLOB_data, in);
    ex.input(embed_param_id::BLOB_data, in);
//    ex.input(mobilefacenet_v2_param_id::BLOB_data, in);

    // get output
    ncnn::Mat out;
//    ex.extract(arcface_param_id::BLOB_fc1, out);
    ex.extract(embed_param_id::BLOB_fc1, out);
//    ex.extract(mobilefacenet_v2_param_id::BLOB_fc1, out);

    // char out_w[80];
    // sprintf(out_w, "%d", out.w);
    // printf ("out w is [%s] \n", out_w);
    std::vector<float> vec;
    for (int i = 0; i < out.w; ++i) {
        vec.emplace_back(*(static_cast<float*>(out.data) + i));
    }

    return vec;
}

float Embedder::get_l2_similarity(
        std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != v2.size())
        throw std::invalid_argument("Wrong size");

    double mul = 0;
    for (int i = 0; i < v1.size(); ++i) {
        mul += v1[i] * v2[i];
    }

    if (mul < 0) {
        return 0;
    }

    return 1 - mul;
}

float Embedder::cosine_distance(std::vector<float> A, std::vector<float>B)
{
    float mul = 0.0;
    float d_a = 0.0;
    float d_b = 0.0;

    if (A.size() != B.size())
    {
        throw std::logic_error("Vector A and Vector B are not the same size");
    }

    // Prevent Division by zero
    if (A.size() < 1)
    {
        throw std::logic_error("Vector A and Vector B are empty");
    }

    std::vector<float>::iterator B_iter = B.begin();
    std::vector<float>::iterator A_iter = A.begin();
    for( ; A_iter != A.end(); A_iter++ , B_iter++ )
    {
        mul += *A_iter * *B_iter;
        d_a += *A_iter * *A_iter;
        d_b += *B_iter * *B_iter;
    }

    if (d_a == 0.0f || d_b == 0.0f)
    {
        throw std::logic_error(
                "cosine similarity is not defined whenever one or both "
                "input vectors are zero-vectors.");
    }

    return 1 - mul / (sqrt(d_a) * sqrt(d_b));
}

void Embedder::normalize(std::vector<float>& arr)
{
    double mod = 0.0;

    for (float i : arr) {
        mod += i * i;
    }

    double mag = sqrt(mod);

    char embed_size[80];

    if (mag == 0) {
        throw std::logic_error("The input vector is a zero vector");
    }

    for (float & i : arr) {
        i /= mag;
    }
}

inline void Embedder::SetDefaultParams(){
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;
}

