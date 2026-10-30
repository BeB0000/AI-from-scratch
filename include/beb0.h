#ifndef BEBO_H
#define BEBO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>

#define HIDDEN_DIM 64
#define NUM_HEADS 4
#define NUM_LAYERS 2
#define VOCAB_SIZE 16000
#define FF_MULTIPLIER 4
#define MAX_SEQ_LEN 64
#define HEAD_DIM (HIDDEN_DIM / NUM_HEADS)
#define FF_DIM (FF_MULTIPLIER * HIDDEN_DIM)

#define LEARNING_RATE 3.0e-4f
#define FIRE_LEARNING_RATE 1e-5f
#define MIN_LEARNING_RATE 1e-4f
#define PHYSICAL_BATCH 1
#define GRAD_ACCUM_STEPS 4
#define LOGICAL_BATCH (PHYSICAL_BATCH * GRAD_ACCUM_STEPS)
#define NUM_STEPS 50000
#define SAVE_EVERY 1000
#define LOG_EVERY 100

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f
#define CLIP_GRAD_NORM 1.0f
#define LN_EPS 1e-5f
#define GELU_C 0.044715f
#define GELU_SQ 0.7978845608f

#define TOKEN_UNK 0
#define TOKEN_BOS 1
#define TOKEN_EOS 2
#define TOKEN_PAD 3

#define THERMAL_PATH "/sys/class/thermal/thermal_zone0/temp"
#define THERMAL_WARN_C 75
#define THERMAL_CRIT_C 85
#define THERMAL_SLEEP_US 300000

#define TOP_K 40
#define REP_PENALTY_WINDOW 10
#define REP_PENALTY_FACTOR 0.8f

#endif
