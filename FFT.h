#ifndef FFT_H
#define FFT_H

#include <deque>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>


class FFT
{

public:
    FFT();
    ~FFT();

public:
	void setNbSignals(int);
	void setFps(float);
	void setBufferedSignalValues(std::vector<std::deque<float>> vBufferedSignalValues);
	std::vector<std::deque<float>> getPowerSpectrum();

private:
    void compute(std::vector<float> &vInputSignal, std::vector<float> &vFFTRealPart, std::vector<float> &vFFTImagPart);
    void transformRadix2(std::vector<float> &real, std::vector<float> &imag);
    std::size_t reverseBits(std::size_t x, unsigned int n);
    void powerSpectrum(std::vector<float> &vFFTRealPart, std::vector<float> &vFFTImagPart, std::vector<float> &vFFTPowerSpectrum);
    void phaseSpectrum(std::vector<float> &vFFTRealPart, std::vector<float> &vFFTImagPart, std::vector<float> &vFFTPowerSpectrum);
    void hannWindow(std::vector<float>& inSignal, std::vector<float>& outSignal);
    void pad(std::vector<float> &vInputSignal, std::vector<float> &vPaddedSignal);
    unsigned long findUpperPowerOfTwo(unsigned long v);

private:
    int m_i32InSignalLength; /**< padded signal (legnth=power of 2) to be processed */
    std::vector<double> m_vf64cosTable; /**< the cosine table  */
    std::vector<double> m_vf64sinTable; /**< the sine table  */
    bool m_bTrigoTablesComputed;/**<  a boolean to check if we need to compute the trigonometric tables  */
    std::vector<std::deque<float>> m_vPowerSpectrumValues;

    int m_i32NbSignals;
    float m_fFps;

};

#endif
