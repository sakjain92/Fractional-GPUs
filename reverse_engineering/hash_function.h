#ifndef __HASH_FUNCTION_H__
#define __HASH_FUNCTION_H__

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

void hash_print_solutions(hash_context_t *ctx);

void hash_del(hash_context *ctx);

#endif /* __HASH_FUNCTION_H__ */
