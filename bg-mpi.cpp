#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

#define BG1 "bg1/frame_"
#define BG2 "bg2/frame_"
#define BG3 "bg3/frame_"
#define BG4 "bg4/frame_"
#define BG5 "bg5/frame_"
#define BG6 "bg6/frame_"
#define BG7 "bg7/frame_"
#define BG1_S 0
#define BG1_E 77
#define BG2_S 78
#define BG2_E 130
#define BG3_S 145
#define BG3_E 188
#define BG4_S 189
#define BG4_E 246
#define BG5_S 380
#define BG5_E 530
#define BG6_S 0
#define BG6_E 150
#define BG7_S 0
#define BG7_E 150
#define FG1 "bg1/foreground.png"
#define FG2 "bg2/foreground.png"
#define FG3 "bg3/foreground.png"
#define FG4 "bg4/foreground.png"
#define FG5 "bg5/foreground.png"
#define FG6 "bg6/foreground.png"
#define FG7 "bg7/foreground.png"
#define THR 40

using namespace std;
using namespace cv;


int loadImages(const string& pathPrefix, int start, int end, vector<Mat>& frames, int rank, int size);
void average_image(Mat & avg_bg, vector<Mat> & bg, int rows, int cols, int channels, int size);
void foreground_mask(Mat & foreground, Mat & fg,  Mat & avg_bg, int rows, int cols, int channels );

int main() {


    MPI_Init(NULL, NULL);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    vector<Mat> bg1, bg2, bg3, bg4, bg5;


    auto start = chrono::high_resolution_clock::now();
    int length1 = loadImages(BG7, BG7_S, BG7_E, bg1, rank, size);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Loaded images successfully. from rank " << rank << " in time "<< duration.count() << " seconds!" << endl;

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




    Mat avg_bg1_final(rows, cols, type, ch);
;
    vector<Mat> avg_bgs = {avg_bg1_final};

    start = chrono::high_resolution_clock::now();
    average_image(avg_bg1, bg1, rows, cols, channels, size);

    MPI_Allreduce(avg_bg1.data, avg_bg1_final.data, rows * cols * channels, MPI_UNSIGNED_CHAR, MPI_SUM, MPI_COMM_WORLD);

    end = chrono::high_resolution_clock::now();
    duration = end - start;

    cout << "Averaged images successfully. from rank " << rank << " in time "<< duration.count() << " seconds!" << endl;




    if(rank == 0) {
        for(int i = 0; i < 1; i++) {
            string filepath = "Output/bg_mpi" + to_string(i+1) + ".png";
            imwrite(filepath, avg_bgs[i]);
        }
    }

    Mat fg1(rows, cols, CV_8UC1, Scalar(0));


    Mat fg1_final(rows, cols, CV_8UC1, Scalar(0));


    vector<Mat> fgs = {fg1_final};

    Mat foreground7 = imread(FG7, IMREAD_GRAYSCALE);

    start = chrono::high_resolution_clock::now();
    foreground_mask(foreground7, fg1, avg_bg1_final, rows, cols, channels);


    MPI_Reduce(fg1.data, fg1_final.data, cols * rows, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);

    end = chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "Foregrounded images successfully. from rank " << rank << " in time "<< duration.count() << " seconds!" << endl;




    if(rank == 0) {
        for(int i = 0; i < 1; i++) {
            string filepath = "Output/fg_mpi" + to_string(i+1) + ".png";
            imwrite(filepath, fgs[i]);
        }
    }



    MPI_Finalize();
    return 0;

}


int loadImages(const string& pathPrefix, int start, int end, vector<Mat>& frames, int rank, int size) {

    int totalFrames = end - start + 1;
    int framesPerProcess = totalFrames / size;
    int startIdx = start + rank * framesPerProcess;
    int endIdx = (rank == size - 1) ? end : startIdx + framesPerProcess - 1;

    double weight = static_cast<double>(endIdx - startIdx) / static_cast<double>(size);

    for(int i = startIdx; i <= endIdx; i++) {
        string filename = pathPrefix + to_string(i) + ".png";
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if(!img.empty()){
            frames.push_back(img);
        }
    }

    return frames.size();

}

void average_image(Mat & avg_bg, vector<Mat> & bg, int rows, int cols, int channels, int size) {

    int n = bg.size();
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            if ( channels ==  1) {
                int sumGray = 0;

                for( int k = 0; k < bg.size(); k++ ) {
                    uchar pixel = bg[k].at<uchar>(y,x);
                    sumGray += pixel;
                }

                float avgGray = static_cast<float>(sumGray) / (n * size);
                avg_bg.at<uchar>(y, x) = static_cast<uchar>(round(avgGray));

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
            }
        }
    }


}
