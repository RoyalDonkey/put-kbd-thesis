#ifndef MICROPHONE_H
#define MICROPHONE_H

#include <SDL2/SDL.h>
#include "tinywav/tinywav.h"

int microphone_init(const char *fpath_wav);
void microphone_record(int enabled);
void microphone_free(void);

#endif /* MICROPHONE_H */
