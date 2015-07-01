/*
 * DualToneMultiFrequencySignalingDataFunction.cpp
 *
 *  Created on: Jun 19, 2015
 *      Author: sam
 */

#include "DualToneMultiFrequencySignalingDataFunction.h"
#include <fstream>
#include <random>
#include <algorithm>
/* Search the DTMF tone bins and find the closest dtmf (hi and lo) that match.
 * Returns the DTMF key that generates the two tones together.
 */
#define DTMF_BAD_DIGIT -1 /* -1 isnt a key :-)    */

#define DEBUG_PLAYSOUND

DualToneMultiFrequencySignalingDataFunction::DualToneMultiFrequencySignalingDataFunction() :
    format(SND_PCM_FORMAT_S16_LE), rate(48000), max_phase(2.0f * M_PI), channels(1), //
    duration_ms(20.0f), numberOfRepeats(2), //
    keyLength(ceil(rate * (duration_ms / 1000.0f) * numberOfRepeats)), //
    asoundLib(new AsoundLib(rate, channels, //
        (keyLength / 10 / numberOfRepeats), SND_PCM_FORMAT_S16_LE))
{
}

DualToneMultiFrequencySignalingDataFunction::~DualToneMultiFrequencySignalingDataFunction()
{
  delete asoundLib;

  for (auto iter = dtmfDataVector.begin(); iter != dtmfDataVector.end(); ++iter)
    delete[] *iter;
  dtmfDataVector.clear();
}

void DualToneMultiFrequencySignalingDataFunction::configure(
    const ConfigurationDescription* configuration)
{
  asoundLib->configure();

  /* https://en.wikipedia.org/wiki/Dual-tone_multi-frequency_signaling */
  dtmfVector.push_back(std::make_pair(941.0f, 1336.0f)); /* 0 */
  dtmfVector.push_back(std::make_pair(697.0f, 1209.0f)); /* 1 */
  dtmfVector.push_back(std::make_pair(697.0f, 1336.0f)); /* 2 */
  dtmfVector.push_back(std::make_pair(697.0f, 1477.0f)); /* 3 */

  dtmfVector.push_back(std::make_pair(770.0f, 1209.0f)); /* 4 */
  dtmfVector.push_back(std::make_pair(770.0f, 1336.0f)); /* 5 */
  dtmfVector.push_back(std::make_pair(770.0f, 1477.0f)); /* 6 */

  dtmfVector.push_back(std::make_pair(852.0f, 1209.0f)); /* 7 */
  dtmfVector.push_back(std::make_pair(852.0f, 1336.0f)); /* 8 */
  dtmfVector.push_back(std::make_pair(852.0f, 1477.0f)); /* 9 */

  dtmfVector.push_back(std::make_pair(697.0f, 1633.0f)); /* A */
  dtmfVector.push_back(std::make_pair(770.0f, 1633.0f)); /* B */
  dtmfVector.push_back(std::make_pair(852.0f, 1633.0f)); /* C */
  dtmfVector.push_back(std::make_pair(941.0f, 1633.0f)); /* D */

  dtmfVector.push_back(std::make_pair(941.0f, 1209.0f)); /* * */
  dtmfVector.push_back(std::make_pair(941.0f, 1477.0f)); /* # */

  for (size_t i = 0; i < dtmfVector.size(); ++i)
  {
    short* buf = new short[keyLength];
    makeDTMF(dtmfVector[i].first, dtmfVector[i].second, buf, keyLength);
    dtmfDataVector.push_back(buf);
  }

  std::string fname = "testing.pcm";

  // Lets try some stuff
  std::vector<int> keys = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
//  std::vector<int> keys = { 0, 1, 2 };
//  genTone(keys, fname.c_str());

  analyizeTone(fname.c_str());

}

void DualToneMultiFrequencySignalingDataFunction::makeDTMF(const float& freqLo, const float& freqHi,
    short* buf, const size_t& bufLen)
{
  /* Tone generator: DTMF
   * RATE : Samples per second
   * buf  : All the samples needed to supply audio for 'secs' seconds
   * a    : Low DTMF tone at point 'i'
   * b    : High DTMF tone at point 'i'
   *
   * tone at sample 'i' = sin(i * ((2*PI) * freq/RATE));
   *
   * More info at:
   * https://stackoverflow.com/questions/1399501/generate-dtmf-tones
   */

  for (size_t i = 0; i < bufLen; ++i)
    buf[i] = (sin(max_phase * freqLo / rate * i) + sin(max_phase * freqHi / rate * i)) * 16383;

}

int DualToneMultiFrequencySignalingDataFunction::findDTMFDigit(fftw_complex *fft)
{
  /* Find match */
  int i = 0;
  for (auto& x : dtmfVector)
  {
    int lo_idx = (int) x.first;
    int hi_idx = (int) x.second;
    double lo = fabs(fft[lo_idx][1]);
    double hi = fabs(fft[hi_idx][1]);
    if (lo > 1.0 && hi > 1.0)
      return i;
    ++i;
  };
  return DTMF_BAD_DIGIT;
}

/* Returns true if 'tone' is found */
bool DualToneMultiFrequencySignalingDataFunction::findTone(fftw_complex *fft)
{
  /* Take the FFT index (bin) and convert back to a frequency
   * The freq represented by each bin is:
   * freq = index * sample_rate / num_samples
   * there are num_samples indexes or bins.
   *
   * Thanks to:
   * https://stackoverflow.com/questions/4364823/how-to-get-frequency-from-fft-result
   */
  int digit = findDTMFDigit(fft);
  if (digit != DTMF_BAD_DIGIT)
  {
    printf("==> DTMF Key %d <==\n", digit);
    return true;
  }
  return false;
}

void DualToneMultiFrequencySignalingDataFunction::genTone(const std::vector<int>& tones,
    const std::string& fname)
{
  std::ofstream fp(fname.c_str(), std::ios::binary | std::ios::out);
  if (!fp.is_open())
    ERR("Could not locate an output file or stdout to write to");

  for (auto& tone : tones)
    fp.write((const char*) dtmfDataVector[tone], keyLength * sizeof(short));

  fp.close();

  for (auto& tone : tones)
    asoundLib->playback(dtmfDataVector[tone], keyLength);
}

void DualToneMultiFrequencySignalingDataFunction::analyizeTone(const std::string& fname)
{
  std::ifstream fp(fname.c_str(), std::ios::binary | std::ios::in);
  size_t bufLen = 0;
  double* buf = nullptr;

  if (fp.is_open())
  {
    // get the length of the file
    fp.seekg(0, fp.end);
    size_t length = fp.tellg();
    fp.seekg(0, fp.beg);

    char* pi8buf = new char[length];
    fp.read(pi8buf, length);

    short* pi16buf = (short*) pi8buf; //LE
    bufLen = length / sizeof(short);
    buf = (double*) fftw_malloc(sizeof(double) * bufLen);
    std::copy(pi16buf, pi16buf + bufLen, buf);
    delete[] pi8buf;
    fp.close();
  }

  if (buf)
  {
    assert(bufLen % keyLength == 0);

    const std::string fftfname = "fft.txt";
    if (remove(fftfname.c_str()) != 0)
      std::cerr << "removal failed" << std::endl;
    else
      std::cout << "file removed" << std::endl;
    std::ofstream fp2(fftfname, std::ios_base::app | std::ios_base::out);

    std::random_device rd;
    std::mt19937 gen(rd());

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution<> d(0, 500.0f);

    size_t ptr = 0;
    const size_t offset = keyLength * (0.0f / 100.0f);
    size_t start_ptr = offset;

    //Process data
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * keyLength);

    const size_t sample_shift = (keyLength / 2 - 1) * rate / keyLength;

    std::cout << "sample_shift: " << sample_shift << std::endl;

    while (start_ptr < bufLen)
    {
      if (start_ptr + keyLength > bufLen)
      {
        std::cout << "OVER RUN!" << std::endl;
        break;
      }
      // Hann function
      const double hannConstant = max_phase / (keyLength - 1);
      for (size_t i = 0; i < keyLength; i++)
      {
        *(buf + start_ptr + i) += d(gen);
        *(buf + start_ptr + i) *= 0.5f * (1.0f - cos(i * hannConstant));
      }

      fftw_plan plan = fftw_plan_dft_r2c_1d(keyLength, (buf + start_ptr), out, FFTW_ESTIMATE);
      fftw_execute(plan);

      //Read the results
      //if (findTone(out))
      //  printf("==> Matched \n");

      // verbose
      size_t nc = (keyLength / 2 + 1);

      /*std::cout << "Output FFT Coefficients:" << std::endl;

       for (size_t i = 0; i < nc; i++)
       printf("  %4d  %12f  %12f\n", i, out[i][0], out[i][1]);
       */

      //fprintf(stdout, "     Frequency  Real       Imag        Abs       Power\n");
      std::cout << "ptr: " << ptr << std::endl;

      for (size_t idx = 0; idx < nc; idx++)
      {
        double realVal = out[idx][0] * 2.0f / keyLength;
        double imagVal = out[idx][1] * 2.0f / keyLength;
        double processed = std::pow(realVal, 2) + std::pow(imagVal, 2);
        //double absVal = sqrt(powVal / 2);
        //if (idx == 0)
        //{
        //  powVal /= 2;
        // }
        //fprintf(stdout, "%10i %10.4lf %10.4lf %10.4lf %10.4lf\n", idx, realVal, imagVal, absVal,
        //    powVal);
        processed = 10. / log(10.0f) * log(processed + 1e-6);   // dB

        // The resulting spectral values in 'processed' are in dB and related to a maximum
        // value of about 96dB. Normalization to a value range between 0 and 1 can be done
        // in several ways. I would suggest to set values below 0dB to 0dB and divide by 96dB:

        // Transform all dB values to a range between 0 and 1:
        if (processed <= 0)
          processed = 0;
        else
        {
          processed /= 96.0f;             // Reduce the divisor if you prefer darker peaks
          if (processed > 1)
            processed = 1;
        }

        double freq = idx * rate / double(keyLength);
        //if (freq > 400.0f && freq < 2000.0f)
        {
          freq += (ptr * sample_shift);
          fp2 << freq << " " << realVal << " " << imagVal << " " << processed << std::endl;

          // The total signal power of a frequency is the sum of the power of the posive and the negative frequency line.
          // Because only the positive spectrum is calculated, the power is multiplied by two.
          // However, there is only one single line in the prectrum for DC.
          // This means, the DC value must not be doubled.
          fp2.flush();
        }
      }

      ++ptr;
      start_ptr += keyLength;

      /*
       Set up an arrray to hold the backtransformed data IN2,
       get a "plan", and execute the plan to backtransform the OUT
       FFT coefficients to IN2.
       */
      //double* in2 = (double*) fftw_malloc(sizeof(double) * bufLen);
      //fftw_plan plan_backward = fftw_plan_dft_c2r_1d(bufLen, out, in2, FFTW_ESTIMATE);
      //fftw_execute(plan_backward);
      /*printf("\n");
       printf("  Recovered input data divided by N:\n");
       printf("\n");

       for (size_t i = 0; i < bufLen; i++)
       {
       printf("  %4d  %12f\n", i, in2[i] / (double) (bufLen));
       }
       *//*
       Release the memory associated with the plans.
       */

      //Clean up
      fftw_destroy_plan(plan);
    }

    std::cout << "start_ptr: " << start_ptr << " bufLen: " << bufLen << std::endl;
    //assert(start_ptr == bufLen);
    //fftw_destroy_plan(plan_backward);
    fftw_free(out);
    fftw_free(buf);
    //fftw_free(in2);
  }
}
