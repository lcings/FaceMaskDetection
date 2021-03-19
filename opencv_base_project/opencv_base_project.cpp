#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace std;

#define RUN_TYPE_IMAGE        0
#define RUN_TYPE_VIDEO        1

#define USE_RUN_TYPE RUN_TYPE_VIDEO

vector<vector<float>> generate_anchors(const vector<float> &ratios, const vector<int> &scales, vector<float> &anchor_base)
{
    vector<vector<float>> anchors;
    for (int idx = 0; idx < scales.size(); idx++) {
        vector<float> bbox_coords;
        int s = scales[idx];
        vector<float> cxys;
        vector<float> center_tiled;
        for (int i = 0; i < s; i++) {
            float x = (0.5 + i) / s;
            cxys.push_back(x);
        }

        for (int i = 0; i < s; i++) {
            float x = (0.5 + i) / s;
            for (int j = 0; j < s; j++) {
                for (int k = 0; k < 8; k++) {
                    center_tiled.push_back(cxys[j]);
                    center_tiled.push_back(x);
                    //printf("%f %f ", cxys[j], x);
                }
                //printf("\n");
            }
            //printf("\n");
        }

        vector<float> anchor_wh;
        for (int i = 0; i < anchor_base.size(); i++) {
            float scale = anchor_base[i] * pow(2, idx);
            anchor_wh.push_back(-scale / 2.0);
            anchor_wh.push_back(-scale / 2.0);
            anchor_wh.push_back(scale / 2.0);
            anchor_wh.push_back(scale / 2.0);
            //printf("%f %f %f %f\n", -scale / 2.0, -scale / 2.0, scale / 2.0, scale / 2.0);
        }

        for (int i = 0; i < anchor_base.size(); i++) {
            float s1 = anchor_base[0] * pow(2, idx);
            float ratio = ratios[i + 1];
            float w = s1 * sqrt(ratio);
            float h = s1 / sqrt(ratio);
            anchor_wh.push_back(-w / 2.0);
            anchor_wh.push_back(-h / 2.0);
            anchor_wh.push_back(w / 2.0);
            anchor_wh.push_back(h / 2.0);
            //printf("s1:%f, ratio:%f w:%f h:%f\n", s1, ratio, w, h);
            //printf("%f %f %f %f\n", -w / 2.0, -h / 2.0, w / 2.0, h / 2.0);
        }

        int index = 0;
        //printf("\n");
        for (float &a : center_tiled) {
            float c = a + anchor_wh[(index++) % anchor_wh.size()];
            bbox_coords.push_back(c);
            //printf("%f ", c);
        }

        //printf("bbox_coords.size():%d\n", bbox_coords.size());
        int anchors_size = bbox_coords.size() / 4;
        for (int i = 0; i < anchors_size; i++) {
            vector<float> f;
            for (int j = 0; j < 4; j++) {
                f.push_back(bbox_coords[i * 4 + j]);
            }
            anchors.push_back(f);
        }
    }

    return anchors;
}

vector<cv::Rect> decode_bbox(vector<vector<float>> &anchors, float *delta, int img_w, int img_h)
{
    vector<cv::Rect> rects;
    float v[4] = { 0.1, 0.1, 0.2, 0.2 };

    int i = 0;
    for (vector<float>& k : anchors) {
        float acx = (k[0] + k[2]) / 2;
        float acy = (k[1] + k[3]) / 2;
        float acw = (k[2] - k[0]);
        float ach = (k[3] - k[1]);

        float r0 = delta[i++] * v[i % 4];
        float r1 = delta[i++] * v[i % 4];
        float r2 = delta[i++] * v[i % 4];
        float r3 = delta[i++] * v[i % 4];

        float centet_x = r0 * acw + acx;
        float centet_y = r1 * ach + acy;

        float w = exp(r2) * acw;
        float h = exp(r3) * ach;
        float x = (centet_x - w / 2) * img_w;
        float y = (centet_y - h / 2) * img_h;
        w *= img_w;
        h *= img_h;
        rects.push_back(cv::Rect(x, y, w, h));
    }

    return rects;
}

vector<int> mns_cv_dnn(vector<cv::Rect> &rects, float *confidences, int c_len, vector<int> &classes, vector <float>&scores)
{
    vector<int> keep_idxs;
    float conf_thresh = 0.75;
    float iou_thresh = 0.7;
    if (rects.size() <= 0) {
        return keep_idxs;
    }
    
    for (int i = 0; i < c_len; i += 2) {
        float max = confidences[i];
        int classess = 0;
        if (max < confidences[i + 1]) {
            max = confidences[i + 1];
            classess = 1;
        }
        classes.push_back(classess);
        scores.push_back(max);
    }

    cv::dnn::NMSBoxes(rects, scores, conf_thresh, 1.0 - iou_thresh, keep_idxs);
    return keep_idxs;
}

int main()
{
    vector<float> anchor_base = { (float)0.04, (float)0.056 };
    vector<float> ratios = { (float)1.0, (float)0.62, (float)0.42 };
    vector<int> scales = { 33, 17, 9, 5, 3 };
    vector<vector<float>> anchors = generate_anchors(ratios, scales, anchor_base);
    cv::dnn::Net net = cv::dnn::readNetFromCaffe("face_mask_detection.prototxt", "face_mask_detection.caffemodel");

#if (USE_RUN_TYPE == RUN_TYPE_IMAGE)
    cv::Mat img = cv::imread("11.jpg");
    if (!img.data) {
        printf("!img.data\n");
        return 0;
    }
#endif

#if (USE_RUN_TYPE == RUN_TYPE_VIDEO)
    cv::VideoCapture vc("demo.mp4");
    cv::Mat img;
    while (1)
    {
        vc >> img;
        if (!img.data) {
            break;
        }
#endif

        cv::Mat rgb_img;
        cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
        cv::Mat input_blob = cv::dnn::blobFromImage(rgb_img, 1 / 255.0, cv::Size(260, 260), cv::Scalar(0, 0, 0), false);

        net.setInput(input_blob, "data");

        vector< cv::String >  targets_node{ "loc_branch_concat","cls_branch_concat" };
        vector< cv::Mat > targets_blobs;

        net.forward(targets_blobs, targets_node);

        cv::Mat feature_bboxes = targets_blobs[0];
        cv::Mat feature_score = targets_blobs[1];
        float *bboxes = (float*)feature_bboxes.data;
        float *confidences = (float*)feature_score.data;

        vector<cv::Rect> rects = decode_bbox(anchors, bboxes, rgb_img.cols, rgb_img.rows);
        vector<int> classes;
        vector <float> scores;
        vector<int> keep_idxs = mns_cv_dnn(rects, confidences, feature_score.total(), classes, scores);

        for (int i : keep_idxs) {
            char str[64];
            cv::Scalar str_coclr;
            if (classes[i] == 0) {
                snprintf(str, 64, "mask");
                str_coclr = cv::Scalar(0, 255, 255);
            }
            else {
                snprintf(str, 64, "numask");
                str_coclr = cv::Scalar(0, 0, 255);
            }

            cv::Rect &r = rects[i];
            cv::putText(img, str, r.tl(), 1, 1.0, str_coclr);
            snprintf(str, 64, "%0.2f%%", scores[i] * 100);
            cv::putText(img, str, cv::Point(r.x, r.y + 10), 1, 0.8, cv::Scalar(255, 255, 255));
            cv::rectangle(img, r, cv::Scalar(0, 255, 255));
        }

        cv::imshow("img", img);

#if (USE_RUN_TYPE == RUN_TYPE_IMAGE)
        cv::waitKey(0);
#endif

#if (USE_RUN_TYPE == RUN_TYPE_VIDEO)
        if ('q' == cv::waitKey(1)) {
            break;
        }
    }
#endif

    return 0;
}
