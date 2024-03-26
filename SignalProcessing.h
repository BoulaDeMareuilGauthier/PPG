#include <numeric>
#include <deque>
#include "opencv2/opencv.hpp"

float computeTemporalAverage(std::deque<float> vSignal);

float computeTemporalStd(std::deque<float> vSignal);

std::vector<float> normalizeTemporalSignal(std::deque<float> vSignal, float avg, float std);

std::vector<float> computeFourierTransform(std::vector<float> vSignal);

int computeHeartRate(std::vector<float> vPowerSpectrum, float lowFreq, float highFreq, float samplingFreq);
