#include "opencv2/opencv.hpp"
#include <vector>

bool loadFaceDetector(cv::CascadeClassifier& faceDetector, std::string sFilename);

std::vector<cv::Rect> detectFace(cv::CascadeClassifier& faceDetector, cv::Mat& frame);

cv::Rect extractForeheadROI(std::vector<cv::Rect>& faceROI);

cv::Mat plotGraph(std::vector<float>& vals, int ySize, int lowerBoundary =-1, int highBoundary=-1);
