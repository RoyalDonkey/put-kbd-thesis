#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <stdio.h>
#include <stdlib.h>

#define STOPKEY_CODE '\x1b'
#define STOPKEY_SYM  "ESC"

int keyboard_init(void);
void keyboard_listen(size_t no_keys, FILE *fout);
int keyboard_free(void);

#endif /* KEYBOARD_H */
