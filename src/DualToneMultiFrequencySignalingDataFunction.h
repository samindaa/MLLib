/*
 * DualToneMultiFrequencySignalingDataFunction.h
 *
 *  Created on: Jun 19, 2015
 *      Author: sam
 */

#ifndef DUALTONEMULTIFREQUENCYSIGNALINGDATAFUNCTION_H_
#define DUALTONEMULTIFREQUENCYSIGNALINGDATAFUNCTION_H_

#include "DataFunction.h"
#include <vector>
#include <iostream>
//
#include <fftw3.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
#include "AsoundLib.h"

#define ERR(...) \
do {                                      \
    fprintf(stderr, " : " __VA_ARGS__); \
    fputc('\n', stderr);                  \
    exit(EXIT_FAILURE);                   \
} while ( 0 )

class DualToneMultiFrequencySignalingDataFunction: public DataFunction
{
  private:
    typedef std::vector<std::pair<float, float>> DTMFVector;
    typedef std::vector<short*> DTMFDataVector;
    DTMFVector dtmfVector;
    DTMFDataVector dtmfDataVector;

    snd_pcm_format_t format;
    unsigned int rate;
    const float max_phase;
    const int channels;
    const float duration_ms;
    const int numberOfRepeats;
    const size_t keyLength;
    AsoundLib* asoundLib;

  public:
    DualToneMultiFrequencySignalingDataFunction();
    ~DualToneMultiFrequencySignalingDataFunction();

    void configure(Config* config);

  private:
    void makeDTMF(const float& freqLo, const float& freqHi, short* buf, const size_t& bufLen);
    int findDTMFDigit(fftw_complex *fft);
    bool findTone(fftw_complex *fft);
    void genTone(const std::vector<int>& tones, const std::string& fname);
    void analyizeTone(const std::string& fname);
};

#endif /* DUALTONEMULTIFREQUENCYSIGNALINGDATAFUNCTION_H_ */
