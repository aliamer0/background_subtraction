#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <array>
#include <cstring>

#define TARGET_FRAME "bg6/frame_33.png" // Specify the target frame by filename


namespace fs = std::filesystem;

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string outputFolder = "output1/";

    if (rank == 0) {
        // Create output folder if it doesn't exist
        if (!fs::exists(outputFolder)) {
            fs::create_directory(outputFolder);
        }
    }

    // Ensure all ranks wait for folder creation
    MPI_Barrier(MPI_COMM_WORLD);


    double startTime = MPI_Wtime(); // Start timing

    vector<string> framePaths;

    // Load paths only on rank 0
    if (rank == 0) {
        if (!fs::exists("bg6") || !fs::is_directory("bg6")) {
            cerr << "Error: 'bg6' directory not found or is not a directory." << endl;
            MPI_Finalize();
            return 1;
        }

        for (const auto& entry : fs::directory_iterator("bg6")) {
            if (entry.path().extension() == ".png") {
                framePaths.push_back(entry.path().string());
            }
        }

        if (framePaths.empty()) {
            cerr << "Error: No .png files found in the 'bg6' directory." << endl;
            MPI_Finalize();
            return 1;
        }

        sort(framePaths.begin(), framePaths.end());
    }

    // Broadcast number of frames
    int numFrames = 0;
    if (rank == 0) {
        numFrames = static_cast<int>(framePaths.size());

    }
    MPI_Bcast(&numFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// copy into vector of char arrays to enable scattering
    const int pathLength = 256;
    vector<array<char, pathLength>> allPaths;

    if (rank == 0) {
        allPaths.resize(numFrames);
        for (int i = 0; i < numFrames; ++i) {
            strncpy(allPaths[i].data(), framePaths[i].c_str(), pathLength - 1);
            allPaths[i][pathLength - 1] = '\0'; // ensure null-termination
        }
    }

    // Compute frames per rank
    vector<int> sendCounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int count = numFrames / size + (i < (numFrames % size) ? 1 : 0);
        sendCounts[i] = count * pathLength;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    // Local buffer to receive scattered paths
    int localNumFrames = sendCounts[rank] / pathLength;
    vector<array<char, pathLength>> localFramePaths(localNumFrames);

    MPI_Scatterv(rank == 0 ? allPaths[0].data() : nullptr, sendCounts.data(), displs.data(), MPI_CHAR,
        localFramePaths[0].data(), localNumFrames * pathLength, MPI_CHAR,
        0, MPI_COMM_WORLD);


    // Load sample image to determine dimensions
    Mat sample = imread(localFramePaths[0].data(), IMREAD_GRAYSCALE);
    if (sample.empty()) {
        cerr << "Rank " << rank << ": Failed to read sample image." << endl;
        MPI_Finalize();
        return 1;
    }
    int rows = sample.rows;
    int cols = sample.cols;


    
// Compute local background (frame-wise average)
    Mat localSum = Mat::zeros(rows, cols, CV_32FC1);  // Full image size
    for (int i = 0; i < localNumFrames; ++i) {
        Mat img = imread(localFramePaths[i].data(), IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Rank " << rank << ": Failed to read frame " << localFramePaths[i].data() << endl;
            continue;
        }
        Mat img32;
        img.convertTo(img32, CV_32FC1);
        localSum += img32;
    }

    // Global sum will be divided by total number of frames after reduction
    Mat globalSum = Mat::zeros(rows, cols, CV_32FC1);

    MPI_Reduce(localSum.data, globalSum.data, rows * cols, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    Mat fullBackground;
    if (rank == 0) {
        globalSum /= numFrames;  // Final average
        globalSum.convertTo(fullBackground, CV_8UC1);
    }

    // Broadcast background to all
    if (rank != 0) {
        fullBackground = Mat::zeros(rows, cols, CV_8UC1);
    }
    MPI_Bcast(fullBackground.data, rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);




    // All ranks load the same frame
    if (rank == 0) {
        Mat frame = imread(TARGET_FRAME, IMREAD_GRAYSCALE);
        if (frame.empty()) {
            cerr << "Rank 0: Failed to read target frame." << endl;
            MPI_Finalize();
            return 1;
        }

        Mat mask = Mat::zeros(frame.size(), CV_8UC1);

        for (int r = 0; r < frame.rows; ++r) {
            for (int c = 0; c < frame.cols; ++c) {
                uchar pixel1 = frame.at<uchar>(r, c);
                uchar pixel2 = fullBackground.at<uchar>(r, c);
                uchar absDiff = static_cast<uchar>(abs(pixel1 - pixel2));
                mask.at<uchar>(r, c) = (absDiff > 60) ? 255 : 0;
            }
        }

        imwrite(outputFolder + "foreground_mask_parallel.png", mask);
        imwrite(outputFolder + "background.png", fullBackground);
        cout << "Foreground and background images saved in " << outputFolder << endl;
    }



	/*if i want to process all frames, i can use the following code:*/

    // Each process handles a subset of frames for foreground subtraction
    //int baseFrames = numFrames / size;
    //int extra = numFrames % size;

    //int startFrame = rank * baseFrames + min(rank, extra);
    //int endFrame = startFrame + baseFrames + (rank < extra ? 1 : 0);
	

    //for (int i = startFrame; i < endFrame; ++i) {
    //    Mat frame = imread(allPaths[i].data(), IMREAD_GRAYSCALE);
    //    if (frame.empty()) {
    //        cerr << "Rank " << rank << ": Failed to read frame " << i << endl;
    //        continue;
    //    }

    //    /*Mat diff, mask;
    //    absdiff(frame, fullBackground, diff);
    //    threshold(diff, mask, 70, 255, THRESH_BINARY);*/

    //    // Manual absdiff and threshold
    //    Mat diff = Mat::zeros(frame.size(), CV_8UC1);
    //    Mat mask = Mat::zeros(frame.size(), CV_8UC1);

    //    for (int r = 0; r < frame.rows; ++r) {
    //        for (int c = 0; c < frame.cols; ++c) {
    //            uchar pixel1 = frame.at<uchar>(r, c);
    //            uchar pixel2 = fullBackground.at<uchar>(r, c);
    //            uchar absDiff = static_cast<uchar>(abs(pixel1 - pixel2));
    //            diff.at<uchar>(r, c) = absDiff;
    //            mask.at<uchar>(r, c) = (absDiff > 40) ? 255 : 0;
    //        }
    //    }

    //    string outputName = outputFolder + "foreground_mask_" + to_string(i) + ".png";
    //    imwrite(outputName, mask);
    //}
    //cout << "Foreground images are saved" << endl;
   /* cout << "Rank " << rank << " processed frames " << startFrame << " to " << endFrame - 1 << endl;*/

    // Time measurement
    double endTime = MPI_Wtime();
    double localElapsed = endTime - startTime;

    double maxElapsed;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cout << "Rank " << rank << " execution time: " << localElapsed << " seconds" << endl;
    if (rank == 0) {
        cout << "Total (max) execution time across all ranks: " << maxElapsed << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
