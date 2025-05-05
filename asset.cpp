#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <filesystem>  // C++17 for folder management

namespace fs = std::filesystem;

int main() {
    // Open the video file
    cv::VideoCapture cap("Input_video/bg.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Create the output folder if it doesn't exist
    std::string outputFolder = "./frames";
    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    int frameCount = 0;
    cv::Mat frame;

    while (true) {
        cap >> frame;  // Read next frame
        if (frame.empty())
            break;

        // Generate file name for each frame
        std::ostringstream filename;
        filename << outputFolder << "/frame_"  << frameCount << ".jpg";

        // Save the frame as an image file
        cv::imwrite(filename.str(), frame);

        frameCount++;
    }

    std::cout << "Extracted " << frameCount << " frames to " << outputFolder << std::endl;
    cap.release();
    return 0;
}
