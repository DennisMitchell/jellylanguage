CACHE_SIZE  ?= 256
NUM_THREADS ?= $(shell nproc)

CFLAGS += -Wall -Wextra -O2 -pthread
CFLAGS += -DNUM_THREADS=$(NUM_THREADS) -DCACHE_SIZE=$(CACHE_SIZE)

findhash2: findhash2.c
	$(CC) $(CFLAGS) -o findhash2 findhash2.c
