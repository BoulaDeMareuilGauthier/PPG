#include <iostream>
#include <cstdlib>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include "ImageProcessing.h"
#include "SignalProcessing.h"

#define FPS 30.0
#define BUFFER_DURATION 5 // in seconds
#define BUFFER_SHIFT 1 // in seconds
#define DISCARD_DURATION 5
#define CHANNEL_OF_INTEREST 1
# define LOW_FREQ 50.0
#define HIGH_FREQ 150.0
#define NB_OF_SAMPLES 5 // number of tests

//==GLOBAL VARIABLES==/
// face detector
cv::CascadeClassifier faceDetector; // face detector based on Haar Cascade
cv::CascadeClassifier eyeDetector; // eye detector based on Haar Cascade
std::string firstHaarCascadeFilename = "../data/haarcascade_frontalface_alt.xml";
std::string secondHaarCascadeFilename = "../data/haarcascade_eye_tree_eyeglasses.xml";
// video capture
cv::VideoCapture cap; // see https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
int deviceID = 0;             // 0 = open default camera
int apiID = cv::CAP_ANY;      // 0 = autodetect default API
// discard data because of white auto-balancing
bool isDataDiscarded = true;
int countDiscard = 0;
// face detection -> ROI
bool isFaceDetected = false;
std::vector<cv::Rect> faceRectangles;
cv::Mat frame_face;
cv::Mat forehead_frame;
// buffer
bool isInProcess = true;
bool isBufferFull = false;
std::deque<float> faceSignal;
std::deque<float> foreheadSignal;
std::vector<float> faceSignalNormalized;
int heartRateBPM = -1;



// Function to find the biggest rectangle in a vector of cv::Rect objects
cv::Rect findBiggestRectangle(const std::vector<cv::Rect>& rectangles) {
	// Initialize variables to store the index and area of the biggest rectangle
	int biggestIndex = -1;
	int maxArea = 0;

	// Iterate through the rectangles
	for (size_t i = 0; i < rectangles.size(); ++i) {
		int area = rectangles[i].width * rectangles[i].height;
		if (area > maxArea) {
			maxArea = area;
			biggestIndex = i;
		}
	}

	// Return the biggest rectangle
	if (biggestIndex != -1)
		return rectangles[biggestIndex];
	else
		return cv::Rect(); // Return an empty rectangle if no rectangles found
}

// this function calculate the average of the values inside a float vector
double calculateAverage(const std::vector<float>& nums) {
	float sum = 0;
	for (int i = 0; i < nums.size(); i++) {
		sum += nums[i];
	}
	return static_cast<double>(sum) / nums.size();
}

// this function loads eyedetector based on Haar Cascade
cv::CascadeClassifier eyeCascade;
bool loadEyesDetector(cv::CascadeClassifier& faceDetector, std::string sFilename) {
	return eyeCascade.load(sFilename);
}

int main(int argc, char** argv)
{
	//=========INITIALISATION=========//
	float fps = 15.0;
	int buffer_duration = 15;

	if (argc >= 3)
	{
		fps = atof(argv[1]);
		buffer_duration = atoi(argv[2]);
	}

	//==FACE DETECTOR==/
	// 1- Creates a face detector based on Haar Cascade
	bool bIsFaceDetectorLoaded = loadFaceDetector(faceDetector, firstHaarCascadeFilename);
	bool bIsEyeDetectorLoaded = loadEyesDetector(eyeDetector, secondHaarCascadeFilename);
	// 2- Checks if the face detector was successfully created
	if (!bIsFaceDetectorLoaded) {
		return 1;
	}
	if (!bIsEyeDetectorLoaded) {
		return 1;
	}

	//==VIDEOCAPTURE==/
	// 1- Opens selected camera using selected API
	cap.open(deviceID, apiID);

	// 2- Check if the cam was successfully opened
	if (!cap.isOpened())
	{
		std::cerr << "[ERROR] Unable to open camera!" << std::endl;
		return 2;
	}

	//--- GRAB AND WRITE LOOP
	std::cout << "[INFO] Start grabbing images" << std::endl;
	std::cout << "--> Stay still!" << std::endl;
	std::cout << "--> Press any key to terminate" << std::endl;
	int number_of_samples = 0;
	std::vector<float> HrVectorFace;
	std::vector<float> HrVectorForehead;
	while (isInProcess)
	{

		// Discards data while the camera is performing auto white balancing
		while (isDataDiscarded)
		{
			// 1- Creates a matrix to store the image from the cam
			cv::Mat frame;
			// 2- Waits for a new frame from camera and store it into 'frame'
			cap.read(frame);
			// 3- Checks if the frame is not empty
			if (frame.empty())
			{
				std::cerr << "[ERROR] blank frame was grabbed -> skipping it!" << std::endl;
				break;
			}
			// 4- During the 1st frame, display some info on the screen
			if (countDiscard == 0)
			{
				std::cout << "[INFO] Discarding data during " << DISCARD_DURATION << " seconds" << std::endl;
				std::cout << "--> Let the camera perform auto white balance... ";
			}
			// 5- Increments the counter 
			countDiscard++;
			// 6- Checks stopping criterion 
			if (countDiscard == DISCARD_DURATION * fps)
			{
				isDataDiscarded = false;
				std::cout << "DONE!" << std::endl;
			}

			cv::putText(frame, "Discarding images during auto white balancing", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			cv::imshow("Raw img", frame);

			// Waits the necessary amount of time to achieve the specified frame rate
			//std::cout << (1000.0 / fps) << std::endl;
			if (cv::waitKey(1000.0 / fps) >= 0)
				isInProcess = false;
		}


		// Fills a buffer of data before estimating HR
		int previousProcessingPercentage = 0;

		while (!isBufferFull)
		{
			// 1- Creates a matrix to store the image from the cam
			cv::Mat frame;
			// 2- Waits for a new frame from camera and store it into 'frame'
			cap.read(frame);
			// 3- Checks if the frame is not empty
			if (frame.empty())
			{
				std::cerr << "[ERROR] blank frame was grabbed -> skipping it!" << std::endl;
				break;
			}
			// 4- Detects the principal face in the image and determines its location
			cv::Rect forhead_rect;
			while (!isFaceDetected)
			{
				// 4.1- Detect face(s)
				faceRectangles = detectFace(faceDetector, frame);
				// 4.2- Checks if a face was detected
				if (faceRectangles.size() > 0)
				{
					// 5- Find and select the biggest face if several of them
					cv::Rect face;
					if (faceRectangles.size() > 1) {
						face = findBiggestRectangle(faceRectangles);
					}
					else {
						face = faceRectangles[0];
					}
					// 6- Creates a cropped image around the face
					frame_face = frame(face);

					// 7- Creates a cropped image around the forehead from the eyes position.
					//detect eyes in the face image
					std::vector<cv::Rect> eyeROI;
					eyeCascade.detectMultiScale(frame_face, eyeROI, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
					int vector_size = static_cast<int>(eyeROI.size());
					//building forehead rectangle
					if (vector_size >= 2) {
						// Calculate ylow based on the comparison of y-coordinates
						double y1;
						if (static_cast<double>(eyeROI[0].y) > static_cast<double>(eyeROI[vector_size - 1].y)) {
							y1 = static_cast<double>(eyeROI[0].y - eyeROI[0].height);
						}
						else {
							y1 = static_cast<double>(eyeROI[vector_size - 1].y - eyeROI[vector_size - 1].height);
						}

						// Extract xleft, xright, and yhigh
						double x1 = static_cast<double>(eyeROI[0].x);
						double x2 = static_cast<double>(eyeROI[vector_size - 1].x);
						double y2 = static_cast<double>(face.y);

						// Check for non negative values$
						double xleft, xright, ylow, yhigh;
						if (x2 > x1) {
							xleft = x1;
							xright = x2;
						}
						else {
							xleft = x2;
							xright = x1;
						} if (y2 > y1) {
							ylow = y1;
							yhigh = y2;
						}
						else {
							ylow = x2;
							yhigh = x1;
						}
						double d = (xright - xleft) / 4;
						// Create the forehead rectangle
						double final_y;
						final_y = face.y - face.height;
						cv::Rect forhead_rect(static_cast<int>(xleft), static_cast<int>(final_y),
							static_cast<int>((3 * d) * 2), static_cast<int>(3 * d));
						cv::rectangle(frame_face, forhead_rect, cv::Scalar(0, 255, 0), 2); // Green rectangle with thickness 2
						
						// Try to take forehead rectangle and if it failed, take the face
						try {
							forehead_frame = frame_face(forhead_rect);
						}
						catch (const std::exception& e) {
							forehead_frame = frame_face;
						}

					}
					else {
						forehead_frame = frame_face;

					}
					// Changes boolean value now that the face was detected
					isFaceDetected = true;
				}
			}
			isFaceDetected = false;

			// 8- Computes the average value of the face ROI
			cv::Scalar avg_face = mean(frame_face);

			// 9- Computes the average value of the forehead ROI
			cv::Scalar avg_forehead = mean(forehead_frame);

			// 10 - PCA implementation on the forehead frame applied then to the face
			// Split the image into channels
			std::vector<cv::Mat> channels;
			cv::split(forehead_frame, channels);

			// Convert each channel to floating point
			std::vector<cv::Mat> channels_float;
			for (int i = 0; i < channels.size(); ++i) {
				cv::Mat channel_float;
				channels[i].convertTo(channel_float, CV_32F);
				channels_float.push_back(channel_float);
			}

			// Compute PCA for each channel
			std::vector<cv::PCA> pcas;
			for (int i = 0; i < channels_float.size(); ++i) {
				cv::Mat channel_flat = channels_float[i].reshape(1, 1);
				cv::PCA pca(channel_flat, cv::Mat(), cv::PCA::DATA_AS_ROW);
				pcas.push_back(pca);
			}

			// Check if the green channel has higher eigenvalues than the other channels (red=2, green=1, blue=0)
			double green_eigenvalues_sum = cv::sum(pcas[1].eigenvalues)[0]; // Sum of eigenvalues of the green channel
			double red_eigenvalues_sum = cv::sum(pcas[2].eigenvalues)[0]; // Sum of eigenvalues of the red channel
			double blue_eigenvalues_sum = cv::sum(pcas[0].eigenvalues)[0]; // Sum of eigenvalues of the blue channel
			// Get the channel of interest by selecting the one with highest eigenvalues. It takes the green one as default.
			int channel_of_interest;
			if (blue_eigenvalues_sum >= (green_eigenvalues_sum + red_eigenvalues_sum)) {
				channel_of_interest = 0;
			}
			else if (green_eigenvalues_sum >= (blue_eigenvalues_sum + red_eigenvalues_sum)) {
				channel_of_interest = 1;
			}
			else if (red_eigenvalues_sum >= (blue_eigenvalues_sum + green_eigenvalues_sum)) {
				channel_of_interest = 2;
			}
			else {
				channel_of_interest = 1;
			}

			// 11- Saves the average value for the face/forehead for the given channel (red=2, green=1, blue=0)
			faceSignal.push_back(avg_face[channel_of_interest]);
			foreheadSignal.push_back(avg_forehead[channel_of_interest]);

			// 12- Displays the current buffer size in %
			int currentProcessingPercentage = (float)faceSignal.size() / (fps * buffer_duration) * 100;
			if (currentProcessingPercentage != previousProcessingPercentage)
				std::cout << "[INFO] Buffer size = " << currentProcessingPercentage << "%\r";
			previousProcessingPercentage = currentProcessingPercentage;

			// 13- Sets a stopping criterion
			if (faceSignal.size() == fps * buffer_duration)
			{
				isBufferFull = true;
			}

			// 14- Displays images (face, forehad)
			cv::imshow("Face img", frame_face);
			cv::imshow("Forehead img", forehead_frame);

			// 15- Adds face rectangle on raw image
			cv::rectangle(frame, faceRectangles[0], cv::Scalar(0, 0, 255), 1, 1, 0);
			// 16- Adds forehead rectangle on raw image
			cv::rectangle(frame_face, forhead_rect, cv::Scalar(0, 255, 0), 2); // Green rectangle with thickness 2
			// 17- Adds instructions and results on raw image
			std::string str2Display = "Heart rate: " + std::to_string(heartRateBPM);
			if (heartRateBPM != -1)
				cv::putText(frame, str2Display, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			else
				cv::putText(frame, "Estimating HR, Stay still!", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			// 18- Displays images raw
			cv::imshow("Raw img", frame);

			// Waits the necessary amount of time to achieve the specified frame rate
			if (cv::waitKey(1000.0 / fps) >= 0)
				isInProcess = false;
		}

		// Computes avg and std
		float avg_face = computeTemporalAverage(faceSignal);
		float avg_forehead = computeTemporalAverage(foreheadSignal);
		float std_face = computeTemporalStd(faceSignal);
		float std_forehead = computeTemporalStd(foreheadSignal);

		// Normalizes the temporal signal
		std::vector<float> faceSignalNormalized = normalizeTemporalSignal(faceSignal, avg_face, std_face);
		std::vector<float> foreheadSignalNormalized = normalizeTemporalSignal(foreheadSignal, avg_forehead, std_forehead);

		// Displays normalized signal
		int yRange = fps * buffer_duration;
		cv::imshow("Face signal", plotGraph(faceSignalNormalized, yRange));
		cv::imshow("Forehead signal", plotGraph(foreheadSignalNormalized, yRange));

		// Computes the power spectrum of the normalized signals
		std::vector<float> powerSpectrum_face = computeFourierTransform(faceSignalNormalized);
		std::vector<float> powerSpectrum_forehead = computeFourierTransform(foreheadSignalNormalized);

		// Displays Fourier transform
		cv::imshow("Face Fourier", plotGraph(powerSpectrum_face, yRange, LOW_FREQ / 60 * powerSpectrum_face.size() / fps, HIGH_FREQ / 60 * powerSpectrum_face.size() / fps));
		cv::imshow("Forehead Fourier", plotGraph(powerSpectrum_forehead, yRange, LOW_FREQ / 60 * powerSpectrum_forehead.size() / fps, HIGH_FREQ / 60 * powerSpectrum_forehead.size() / fps));

		// Determines the maximum of the power spectrum in a given frequency range
		// Deduces the heart rate value
		int hr_face = computeHeartRate(powerSpectrum_face, LOW_FREQ, HIGH_FREQ, fps);
		int hr_forehead = computeHeartRate(powerSpectrum_forehead, LOW_FREQ, HIGH_FREQ, fps);

		//std::cout << "HR_face = " << hr_face  << std::endl;
		//std::cout << "HR_forehead = " << hr_forehead  << std::endl;

		// Removes data at the beginning of the deque to update the buffer
		for (int l = 0; l < BUFFER_SHIFT * fps; l++)
		{
			faceSignal.pop_front();
			foreheadSignal.pop_front();
		}
		isBufferFull = false;

		// Waits the necessary amount of time to achieve the specified frame rate
		if (cv::waitKey(1000.0 / fps) >= 0)
			isInProcess = false;

		// Counts the current nb of samples and check if not longer than the test duration
		number_of_samples++;
		std::cout << "nb of samples " << number_of_samples << std::endl;
		if (number_of_samples >= NB_OF_SAMPLES) {
			isInProcess = false;
		}
		HrVectorFace.push_back(static_cast<float>(hr_face));
		HrVectorForehead.push_back(static_cast<float>(hr_forehead));
	}

	// At the end of the measurements, calculate the average HR over the time of measure
	double avgface = calculateAverage(HrVectorFace);
	std::cout << "The average HR for the face is: " << avgface << std::endl;
	double avgforehead = calculateAverage(HrVectorForehead);
	std::cout << "The average HR for the forehead is: " << avgforehead << std::endl;

	// Destroys all GUI
	cv::destroyAllWindows();

	// Releases the camera
	cap.release();

	return 0;
}
