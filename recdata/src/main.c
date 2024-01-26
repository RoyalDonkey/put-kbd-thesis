#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <signal.h>
#include "microphone.h"
#include "keyboard.h"
#include "utils.h"

#define MAX_FNAME_SIZE 1024

int main(int argc, char **argv);
void usage(void);
int parse_cmdline_args(int argc, char **argv);
void sighandler(int signal);
void cleanup(void);

static size_t no_keys;
static char fpath_wav[MAX_FNAME_SIZE],
            fpath_keys[MAX_FNAME_SIZE];
static FILE *fp_keys;


int main(int argc, char **argv) {
	errno = 0;
	signal(SIGINT,  sighandler);
	signal(SIGQUIT, sighandler);
	signal(SIGTERM, sighandler);
	signal(SIGABRT, sighandler);

	if (0 != parse_cmdline_args(argc, argv))
		return 1;
	if (0 != microphone_init(fpath_wav))
		return 1;
	if (0 != keyboard_init())
		return 1;

	fp_keys = fopen(fpath_keys, "w");
	if (fp_keys == NULL) {
		perror("recdata: failed to open keys file");
		return 1;
	}

	microphone_record(1);
	if (no_keys == 0)
		printf("\nrecording started, type now and press "STOPKEY_SYM" when finished...\n");
	else
		printf("\nrecording started, waiting for %zu inputs...\n", no_keys);
	keyboard_listen(no_keys, fp_keys);

	putchar('\n');
	cleanup();
	return 0;
}

void usage(void)
{
	puts("Usage:");
	puts("    recdata [-h, --help] NO_KEYS FNAME");
	puts("");
	puts("    NO_KEYS   Number of keystrokes to register.");
	puts("              When 0, accepts input indefinitely.");
	puts("              Input can be stopped anytime with "STOPKEY_SYM);
	puts("");
	puts("      FNAME   Output filename, extension-less.");
	puts("              Two files will be created, FNAME.wav and FNAME.keys");
}

int parse_cmdline_args(int argc, char **argv)
{
	const char *fname;
	size_t fname_len;

	if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
		usage();
		exit(0);
	}
	if (argc != 3) {
		puts("recdata: invalid arguments");
		usage();
		return 1;
	}

	no_keys = strtoul(argv[1], NULL, 10);
	if (errno) {
		perror("recdata: failed to interpret first argument as unsigned int");
		return 1;
	}

	fname = argv[2];
	fname_len = strnlen(fname, MAX_FNAME_SIZE - 5); /* subtract 5 for ".keys" */
	if (fname_len == MAX_FNAME_SIZE) {
		printf("recdata: maximum filename length exceeded\n");
		return 1;
	}
	strcpy(fpath_wav, fname);
	strcpy(fpath_wav + fname_len, ".wav");
	strcpy(fpath_keys, fname);
	strcpy(fpath_keys + fname_len, ".keys");

	return 0;
}

void sighandler(int signal)
{
	putchar('\n');
	switch(signal) {
		case SIGINT:  printf("received SIGINT!\n");  break;
		case SIGQUIT: printf("received SIGQUIT!\n"); break;
		case SIGTERM: printf("received SIGTERM!\n"); break;
		case SIGABRT: printf("received SIGABRT!\n"); break;
		default:      printf("received signal %d!\n", signal); break;
	}
	if (fp_keys != NULL)
		cleanup();
}

void cleanup(void)
{
	printf("keep quiet, saving files... ");
	fflush(stdout);
	microphone_free();
	keyboard_free();
	fclose(fp_keys);
	printf("done.\n");
}
