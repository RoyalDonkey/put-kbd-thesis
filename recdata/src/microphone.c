#include "microphone.h"
#include <stdio.h>
#include <SDL2/SDL.h>
#include "tinywav/tinywav.h"

#define WAV_NUM_CHANNELS 1
#define WAV_SAMPLE_RATE 44100
#define WAV_SAMPLES 4096

int select_mic(void);
void mic_callback(void *userdata, Uint8 *stream, int len);

static TinyWav tw;
static SDL_AudioDeviceID mic;


int microphone_init(const char *fpath_wav)
{
	if (SDL_Init(SDL_INIT_AUDIO) < 0) {
		printf("recdata: SDL init error: %s\n", SDL_GetError());
		return 1;
	}

	if (select_mic() != 0)
		return 1;

	tinywav_open_write(&tw,
	                   WAV_NUM_CHANNELS,
	                   WAV_SAMPLE_RATE,
	                   TW_INT16,
	                   TW_INLINE,
	                   fpath_wav);
	return 0;
}

void microphone_record(int enabled)
{
	SDL_PauseAudioDevice(mic, !enabled);
}

void microphone_free(void)
{
	SDL_Delay(700); /* Let the final keystroke be heard */
	SDL_CloseAudioDevice(mic);
	tinywav_close_write(&tw);
}

int select_mic(void)
{
	const char *mic_name;
	SDL_AudioSpec want, have;
	SDL_zero(want);
	want.freq = WAV_SAMPLE_RATE;
	/* recdata outputs files in sint16 format. The reason we ask SDL
	 * for float32 here is because tinywav_write_f only accepts
	 * float32-encoded raw audio data as input, even when writing sint16
	 * output files (see tinywav/README.md).  */
	want.format = AUDIO_F32SYS;
	want.channels = WAV_NUM_CHANNELS;
	want.samples = WAV_SAMPLES;
	want.callback = mic_callback;

	const int no_mics = SDL_GetNumAudioDevices(1);
	if (no_mics <= 0) {
		printf("no microphone detected, aborting\n");
		return 1;
	} else {
		printf("found %d microphone%s:\n", no_mics, no_mics > 1 ? "s" : "");
		for (int i = 0; i < no_mics; i++) {
			mic_name = SDL_GetAudioDeviceName(i, 1);
			printf("(%d) %s\n", i+1, mic_name);
		}

		printf("choose: ");
		int index;
		const int err = scanf("%d", &index);
		if (err == EOF) {
			perror("recdata: failed to read input");
			return 1;
		}
		if (err != 1) {
			printf("invalid input\n");
			return 1;
		}
		if (index < 1 || index > no_mics) {
			printf("index out of bounds\n");
			return 1;
		}
		mic_name = SDL_GetAudioDeviceName(index - 1, 1);
	}

	mic = SDL_OpenAudioDevice(mic_name, 1, &want, &have, 0);
	if (mic == 0) {
		SDL_Log("failed to open microphone: %s\n", SDL_GetError());
		return 1;
	}
	if (have.format != want.format) {
		SDL_Log("the selected microphone is not suitable for recording");
		return 1;
	}
	if (no_mics > 1)
		printf("selected %s\n", mic_name);

	return 0;
}

void mic_callback(void *userdata, Uint8 *stream, int len) {
	tinywav_write_f(&tw, stream, len / sizeof(float));
}
