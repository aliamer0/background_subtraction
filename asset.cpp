#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {

    cv::VideoCapture cap("Input_video/input2.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }


    std::string outputFolder = "./frames";
    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    int frameCount = 0;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        std::ostringstream filename;
        filename << outputFolder << "/frame_"  << frameCount << ".png";


        cv::imwrite(filename.str(), frame);

        frameCount++;
    }

    std::cout << "Extracted " << frameCount << " frames to " << outputFolder << std::endl;
    cap.release();
    return 0;
}
