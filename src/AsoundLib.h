/*
 * AsoundLib.h
 *
 *  Created on: Jun 20, 2015
 *      Author: sam
 */

#ifndef ASOUNDLIB_H_
#define ASOUNDLIB_H_

#include <iostream>
#include "alsa/asoundlib.h"

#define CHECK(f) do { \
  int err = (f); \
  if (err < 0) { \
    fprintf(stderr, "%s failed: %s\n", #f, snd_strerror(err)); \
    exit(EXIT_FAILURE); \
  } \
} while (0);

class AsoundLib
{
  private:
    unsigned int rate;
    int channels;
    snd_pcm_uframes_t period;
    // hw
    snd_pcm_t *pin, *pout;
    snd_pcm_hw_params_t *hw;
    snd_pcm_sw_params_t *sw;
    snd_pcm_uframes_t boundary;
    snd_pcm_uframes_t frames;
    size_t buf_size;
    snd_pcm_format_t format;

  public:
    AsoundLib(const unsigned int& rate, const int& channels, const snd_pcm_uframes_t& period,
        const snd_pcm_format_t& format);
    ~AsoundLib();
    void configure();
    void playback(const short* buf, const size_t& bufLen);
};

#endif /* ASOUNDLIB_H_ */
