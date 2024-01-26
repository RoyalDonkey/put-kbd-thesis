#include "keyboard.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "utils.h"

#ifdef __unix__
#include <termios.h>
#include <unistd.h>
static struct termios old_termios;
#elif defined(_WIN32)
#include <conio.h>
#else
#error recdata only supports Unix and Windows operating systems
#endif

static double time_start;

int keyboard_init(void)
{
#ifdef __unix__
	struct termios termios = {0};
	fflush(stdout);
	if (tcgetattr(0, &termios) < 0) {
		perror("recdata: tcgetattr failed");
		return 1;
	}

	old_termios = termios;
	termios.c_lflag &= ~ICANON; /* put terminal in non-canonical mode */
	termios.c_lflag &= ~ECHO;   /* don't echo typed characters */
	termios.c_cc[VMIN] = 1;     /* minimal no. chars expected from each read syscall */
	termios.c_cc[VTIME] = 0;    /* timeout for read syscall (0 == fully blocking) */

	if (tcsetattr(0, TCSANOW, &termios) < 0) {
		perror("recdata: tcsetattr failed");
		return 1;
	}
#endif
	time_start = get_time();
	return 0;
}

#ifdef __unix__
int getch(void)
{
	char ret;
	if (read(0, &ret, sizeof(ret)) != sizeof(ret)) {
		perror("recdata: getch failed");
		exit(1);
	}
	return ret;
}
#endif

void keyboard_listen(size_t no_keys, FILE *fout)
{
	if (no_keys == 0)
		no_keys = SIZE_MAX;
	while (no_keys-- != 0) {
		const char c = getch();
		if (c == EOF) {
			printf("recdata: premature EOF\n");
			return;
		}
		if (c == STOPKEY_CODE) {
			printf("key detected: "STOPKEY_SYM" (0x%02x, %d)\n", STOPKEY_CODE, STOPKEY_CODE);
			printf("stopped by user\n");
			return;
		}
		printf("(%zu left) key detected: '%c' (0x%02x, %d)\n", no_keys, c, c, c);
		fprintf(fout, "%lf %c\n", get_time() - time_start, c);
	}
}

int keyboard_free(void)
{
#ifdef __unix__
	fflush(stdout);
	if (tcsetattr(0, TCSANOW, &old_termios) < 0) {
		perror("recdata: tcsetattr failed");
		return 1;
	}
#endif
	return 0;
}
