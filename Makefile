CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS = -pthread -lm

TARGET = beb0
SRCS = src/beb0.c
OBJS = $(SRCS:.c=.o)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c include/beb0.h
	$(CC) $(CFLAGS) -c $< -o $@

train: $(TARGET)
	./$(TARGET) data/corpus.bin

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS)

clear:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean clear run train
