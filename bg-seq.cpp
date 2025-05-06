#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

#define BG1 "bg1/frame_"
#define BG2 "bg2/frame_"
#define BG3 "bg3/frame_"
#define BG4 "bg4/frame_"
#define BG5 "bg5/frame_"
#define BG1_S 0
#define BG1_E 77
#define BG2_S 78
#define BG2_E 130
#define BG3_S 145
#define BG3_E 188
#define BG4_S 189
#define BG4_E 246
#define BG5_S 247
#define BG5_E 606
#define FG1 "bg1/foreground.jpg"
#define FG2 "bg2/foreground.jpg"
#define FG3 "bg3/foreground.jpg"
#define FG4 "bg4/foreground.jpg"
#define FG5 "bg5/foreground.jpg"
#define THR 90

using namespace std;
using namespace cv;


void loadImages(const string& pathPrefix, int start, int end, vector<Mat>& frames);
void average_image(Mat & avg_bg, vector<Mat> & bg, int rows, int cols, int channels);
void foreground_mask(Mat & foreground, Mat & fg,  Mat & avg_bg, int rows, int cols, int channels );

int main() {




    vector<Mat> bg1, bg2, bg3, bg4, bg5;
    auto start = chrono::high_resolution_clock::now();
    loadImages(BG1, BG1_S, BG1_E, bg1);
    loadImages(BG2, BG2_S, BG2_E, bg2);
    loadImages(BG3, BG3_S, BG3_E, bg3);
    loadImages(BG4, BG4_S, BG4_E, bg4);
    loadImages(BG5, BG5_S, BG5_E, bg5);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Loaded images successfully. in time " << duration.count() << " seconds!" << endl;

    int rows = bg1[0].rows;
    int cols = bg1[0].cols;
    int type = bg1[0].type();
    int channels = bg1[0].channels();

    Scalar ch;
    if (channels == 1) {
        ch = Scalar(0);
    } else if (channels == 3) {
        ch = Scalar(0, 0, 0);
    } else if (channels == 4) {
        ch = Scalar(0, 0, 0, 0);
    }

    Mat avg_bg1(rows, cols, type, ch);
    Mat avg_bg2(rows, cols, type, ch);
    Mat avg_bg3(rows, cols, type, ch);
    Mat avg_bg4(rows, cols, type, ch);
    Mat avg_bg5(rows, cols, type, ch);
    vector<Mat> avg_bgs = {avg_bg1, avg_bg2, avg_bg3, avg_bg4, avg_bg5};

    start = chrono::high_resolution_clock::now();
    average_image(avg_bg1, bg1, rows, cols, channels);
    average_image(avg_bg2, bg2, rows, cols, channels);
    average_image(avg_bg3, bg3, rows, cols, channels);
    average_image(avg_bg4, bg4, rows, cols, channels);
    average_image(avg_bg5, bg5, rows, cols, channels);
    end = chrono::high_resolution_clock::now();
    duration = end - start;

    cout << "Averaged BGs images successfully. in time " << duration.count() << " seconds!" << endl;


    for(int i = 0; i < 5; i++) {
        string filepath = "Output/bg_seq" + to_string(i+1) + ".jpg";
        imwrite(filepath, avg_bgs[i]);
    }

    Mat fg1(rows, cols, CV_8UC1, Scalar(0));
    Mat fg2(rows, cols, CV_8UC1, Scalar(0));
    Mat fg3(rows, cols, CV_8UC1, Scalar(0));
    Mat fg4(rows, cols, CV_8UC1, Scalar(0));
    Mat fg5(rows, cols, CV_8UC1, Scalar(0));
    vector<Mat> fgs = {fg1, fg2, fg3, fg4, fg5};

    Mat foreground1 = imread(FG1, IMREAD_UNCHANGED);
    Mat foreground2 = imread(FG2, IMREAD_UNCHANGED);
    Mat foreground3 = imread(FG3, IMREAD_UNCHANGED);
    Mat foreground4 = imread(FG4, IMREAD_UNCHANGED);
    Mat foreground5 = imread(FG5, IMREAD_UNCHANGED);

    start = chrono::high_resolution_clock::now();
    foreground_mask(foreground1, fg1, avg_bg1, rows, cols, channels);
    foreground_mask(foreground2, fg2, avg_bg2, rows, cols, channels);
    foreground_mask(foreground3, fg3, avg_bg3, rows, cols, channels);
    foreground_mask(foreground4, fg4, avg_bg4, rows, cols, channels);
    foreground_mask(foreground5, fg5, avg_bg5, rows, cols, channels);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "Foregrounded images successfully. in time " << duration.count() << " seconds!" << endl;




    for(int i = 0; i < 5; i++) {
        string filepath = "Output/fg_seq" + to_string(i+1) + ".jpg";
        imwrite(filepath, fgs[i]);
    }

}


void loadImages(const string& pathPrefix, int start, int end, vector<Mat>& frames) {

    for(int i = start; i <= end; i++) {
        string filename = pathPrefix + to_string(i) + ".jpg";
        Mat img = imread(filename, IMREAD_UNCHANGED);
        if(!img.empty()){
            frames.push_back(img);
        }

    } // end of for directive


}

void average_image(Mat & avg_bg, vector<Mat> & bg, int rows, int cols, int channels) {

    int n = bg.size();
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            if ( channels ==  1) {
                int sumGray = 0;

                for( int k = 0; k < bg.size(); k++ ) {
                    uchar pixel = bg[k].at<uchar>(y,x);
                    sumGray += pixel;
                }

                float avgGray = static_cast<float>(sumGray) / n;
                avg_bg.at<uchar>(y, x) = static_cast<uchar>(round(avgGray));

            } else if (channels == 3) {
                int sumR = 0, sumB = 0, sumG = 0;

                for (int k = 0; k < bg.size(); k++) {
                    Vec3b pixel = bg[k].at<Vec3b>(y, x);
                    sumB += pixel[0];
                    sumG += pixel[1];
                    sumR += pixel[2];
                }

               float avgB = static_cast<float>(sumB) / n;
               float avgG = static_cast<float>(sumG) / n;
               float avgR = static_cast<float>(sumR) / n;

               avg_bg.at<Vec3b>(y, x)[0] = static_cast<uchar>(round(avgB));  // Round to nearest integer
               avg_bg.at<Vec3b>(y, x)[1] = static_cast<uchar>(round(avgG));
               avg_bg.at<Vec3b>(y, x)[2] = static_cast<uchar>(round(avgR));

            } else if (channels == 4) {
                 int sumR = 0, sumB = 0, sumG = 0, sumA = 0;

                for (int k = 0; k < bg.size(); k++) {
                    Vec4b pixel = bg[k].at<Vec4b>(y, x);
                    sumB += pixel[0];
                    sumG += pixel[1];
                    sumR += pixel[2];
                    sumA += pixel[3];
                }


                float avgB = static_cast<float>(sumB) / n;
                float avgG = static_cast<float>(sumG) / n;
                float avgR = static_cast<float>(sumR) / n;
                float avgA = static_cast<float>(sumA) / n;

                avg_bg.at<Vec4b>(y, x)[0] = static_cast<uchar>(round(avgB));
                avg_bg.at<Vec4b>(y, x)[1] = static_cast<uchar>(round(avgG));
                avg_bg.at<Vec4b>(y, x)[2] = static_cast<uchar>(round(avgR));
                avg_bg.at<Vec4b>(y, x)[3] = static_cast<uchar>(round(avgA));

            }
        }
    }
}

void foreground_mask(Mat & foreground, Mat & fg,  Mat & avg_bg, int rows, int cols, int channels ) {



    for( int y = 0; y < rows; y++ ) {
        for( int x = 0; x < cols; x++ ) {
            if(channels == 1) {

                int diff = abs(foreground.at<uchar>(y, x) - avg_bg.at<uchar>(y,x));
                if (diff > THR) {
                    fg.at<uchar>(y, x) = 250;
                }
            } else if (channels == 3) {

                int diff1 = abs(foreground.at<Vec3b>(y, x)[0] - avg_bg.at<Vec3b>(y,x)[0]);
                int diff2 =  abs(foreground.at<Vec3b>(y, x)[1] - avg_bg.at<Vec3b>(y,x)[1]);
                int diff3 =  abs(foreground.at<Vec3b>(y, x)[2] - avg_bg.at<Vec3b>(y,x)[2]);
                int diff = diff1 + diff2 + diff3;
                if (diff > (THR * 3)) {
                    fg.at<uchar>(y,x) = 250;
                }
            } else if (channels == 4) {
                int diff1 = abs(foreground.at<Vec4b>(y, x)[0] - avg_bg.at<Vec4b>(y,x)[0]);
                int diff2 =  abs(foreground.at<Vec4b>(y, x)[1] - avg_bg.at<Vec4b>(y,x)[1]);
                int diff3 =  abs(foreground.at<Vec4b>(y, x)[2] - avg_bg.at<Vec4b>(y,x)[2]);
                int diff4 =  abs(foreground.at<Vec4b>(y, x)[3] - avg_bg.at<Vec4b>(y,x)[3]);

                int diff = diff1 + diff2 + diff3;
                if ( diff > (THR * 4) ) {
                    fg.at<uchar>(y, x) = 250;
                }
            }
        }
    }


}
