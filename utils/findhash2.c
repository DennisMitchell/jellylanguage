#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ctz __builtin_ctzll

typedef unsigned long long word_t;
typedef unsigned __int128 dword_t;

const int word_bits = 64;
const int word_bytes = 8;
const int cache_size = CACHE_SIZE;
const int num_threads = NUM_THREADS;

int num_buckets, num_integers;
void *buckets_p, *cache_p, *jumps_p;

volatile word_t score = (word_t) 0 - 1;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void find_hash(int num_buckets, int num_integers, int thread_num)
{
	int *buckets = buckets_p;
	word_t (*cache)[num_integers] = cache_p;
	word_t (*jumps)[num_integers] = jumps_p;

	word_t states[num_integers];
	memset(states, 0, sizeof(states));
	int map[num_buckets];
	word_t hash_num = 0;

	while(hash_num < score)
	{
		for(int i = thread_num; i < cache_size; i += num_threads)
		{
			memset(map, 0, sizeof(map));
			int done = 1;

			for(int j = 0; j < num_integers; j++)
			{
				dword_t state = states[j] + cache[i][j];
				word_t hash = state * num_buckets >> word_bits;

				if(map[hash] && map[hash] != buckets[j])
				{
					done = 0;
					break;
				}

				map[hash] = buckets[j];
			}

			if(done)
			{
				pthread_mutex_lock(&lock);
				hash_num += thread_num;
				score = hash_num < score ? hash_num : score;
				return;
			}

			hash_num += num_threads;
		}

		for(int j = 0, tz = ctz(hash_num); j < num_integers; j++)
			states[j] += jumps[tz][j];
	}
}

void *wrapper(void *thread_num_p)
{
	find_hash(num_buckets, num_integers, * (int *) thread_num_p);

	return NULL;
}

int main(int argc, char *argv[])
{
	assert(argc > 3);

	num_buckets = strtoul(argv[1], NULL, 0);
	assert(num_buckets > 0);

	num_integers = strtoul(argv[2], NULL, 0);
	assert(num_integers > 0 && num_integers == argc - 3);

	int buckets[num_integers];
	buckets_p = buckets;

	for(int i = 0; i < num_integers; i++)
	{
		buckets[i] = strtoul(argv[3 + i], NULL, 0);
		assert(buckets[i] > 0 && buckets[i] <= num_buckets);
	}

	word_t integers[num_integers][word_bits];
	assert(fread(integers, sizeof(integers), 1, stdin) == 1);

	word_t cache[cache_size][num_integers];
	cache_p = cache;
	word_t jumps[word_bits][num_integers];
	jumps_p = jumps;

	for(int i = 0; i < num_integers; i++)
	{
		cache[0][i] = 0;

		for(int j = 1; j < cache_size; j++)
			cache[j][i] = cache[j - 1][i] + integers[i][ctz(j)];

		word_t offset = cache[cache_size - 1][i];

		for(int j = 0; j < word_bits; j++)
			jumps[j][i] = integers[i][j] + offset;
	}

	pthread_t thread_id[num_threads];
	int thread_nums[num_threads];

	for(int i = 0; i < num_threads; i++)
	{
		thread_nums[i] = i;
		pthread_create(&thread_id[i], NULL, wrapper, &thread_nums[i]);
	}

	for(int i = 0; i < num_threads; i++)
		pthread_join(thread_id[i], NULL);

	printf("%llu\n", score - 1);
}
