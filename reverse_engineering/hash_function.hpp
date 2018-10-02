#ifndef __HASH_FUNCTION_HPP__
#define __HASH_FUNCTION_HPP__

/* Forward declarations for external use */
struct hash_context;
typedef struct hash_context hash_context_t;

hash_context_t *hash_init(int min_bit, int max_bit,
        void *start_addr, void *end_addr);

int hash_find_solutions(hash_context_t *ctx, void *arg,
    void *(*find_next_partition_pair)(void *addr1, void *start_addr, 
        void *end_addr, size_t offset, void *arg));

int hash_find_solutions2(hash_context_t *ctx, void *arg,
    bool (*check_partition_pair)(void *addr1, void *addr2, void *arg));

void hash_sort_solutions(hash_context_t *ctx);

void hash_print_solutions(hash_context_t *ctx);

hash_context_t *hash_get_common_solutions(hash_context_t *ctx1, hash_context_t *ctx2);

void *hash_get_next_addr(std::vector<hash_context_t *> ctx, 
        std::vector<int> partitions, void *start_addr, void *end_addr);

bool hash_is_same_partition(hash_context_t *ctx, void *addr1, void *addr2);

#if 0 /* TODO: Below code is just for testing. Remove */
hash_context_t *hash_get_dram(void);
hash_context_t *hash_get_cache(void);
hash_context_t *hash_get_common(void);
#endif

void hash_del(hash_context *ctx);

#endif /* __HASH_FUNCTION_HPP__ */
