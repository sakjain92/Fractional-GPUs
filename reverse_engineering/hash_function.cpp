/* 
 * This file has helper functions to find hash function which is responsible for
 * partitioning in hardware.
 * So reverse engineering happens in following steps:
 *
 * 1) Generate a pair of addresses to test.
 * 2) Test if pair of address lie on same partition.
 * 3) Collect many such pair of addresses.
 * 4) Using brute force, find all hash functions which fit.
 * 5) Repeat till all addresses bits are accounted for.
 *
 * XXX: Currently we only support XOR based hash functions (XORing of physical
 * address bits)
 */
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include <vector>
#include <algorithm>

#include <hash_function.hpp>

/* Maximum physical address bits currently is 64 */
#define MAX_NUM_INDEX        64

typedef struct solution {

    int indexes[MAX_NUM_INDEX];
    int depth;

} solution_t;

typedef struct hash_context {
    int min_bit;
    int max_bit;
    
    uintptr_t start_addr;               /* Range of permissible addresses */
    uintptr_t end_addr;

    std::vector< std::pair <uintptr_t, uintptr_t> > keys;       /* Keys with same hash */
    std::vector<solution_t> solutions;  /* Valid solutions */

} hash_context_t;

static void hash_reduce(hash_context_t *ctx);

static void print_solution(const solution_t &s)
{
    int i;

    for (i = 0; i < s.depth - 1; i++) {
        printf("Bit(%d) ^ ", s.indexes[i]);
    }
    
    if (s.depth > 0)
        printf("Bit(%d)\n", s.indexes[s.depth - 1]);
}


/* Compare equivalence of two solutions */
static int are_solutions_same(const solution_t &a, const solution_t &b)
{
    int i;

    if (a.depth != b.depth)
        return 0;

    for (i = 0; i < a.depth; i++) {
        if (a.indexes[i] != b.indexes[i])
            return 0;
    }

    return 1;
}

/* 
 * Add new bit in a solution.
 * Returns < 0 if can't insert
 */
static int insert_bit_in_solutions(solution_t &s, int bit)
{
    if (s.depth == MAX_NUM_INDEX)
        return -1;

    /* All bits are in ascending order */
    if (s.indexes[s.depth - 1] >= bit)
        return -1;

    s.indexes[s.depth] = bit;
    s.depth++;

    return 0;
}
/* 
 * Create arbitary hash function. Should keep calling this function till
 * it returns negative. In that case, size should be incremented
 */
static int permute_hypothesis(int *array, int size, int min_val, int max_val, int isfirst)
{
    int i, start_index;

    assert(size >= 1);

    if (isfirst) {
        for (i = 0; i < size; i++) {
            array[i] = min_val + i;
        }

        assert(array[size - 1] <= max_val);
        return 0;
    }


    if (array[size - 1] < max_val) {
        array[size - 1]++;
        return 0;
    }

    for (start_index = size - 2; start_index >= 0; start_index--) {
        int valid = 1;
        array[start_index]++;
        for (int i = start_index + 1; i < size; i++) {
            array[i] = array[i - 1] + 1;
            if (array[i] > max_val) {
                valid = 0;
                break;
            }
        }

        if (valid)
            break;
    }

    if (start_index < 0)
        return -1;
    
   return 0;
}

/* Combine two solutions */
static void xor_solutions(solution_t &s1, solution_t &s2, solution_t &r)
{
    int i = 0, j = 0, k = 0;

    memset(&r, 0, sizeof(solution_t));

    while (i < s1.depth || j < s2.depth) {
        if (j == s2.depth) {
            r.indexes[k] = s1.indexes[i];
            i++;
            k++;
        } else if (i == s1.depth) {
            r.indexes[k] = s2.indexes[j];
            j++;
            k++;
        } else if (s1.indexes[i] < s2.indexes[j]) {
            r.indexes[k] = s1.indexes[i];
            i++;
            k++;
        } else if (s1.indexes[i] > s2.indexes[j]) {
            r.indexes[k] = s2.indexes[j];
            j++;
            k++;
        } else {
            assert(s1.indexes[i] == s2.indexes[j]);
            i++;
            j++;
        }
        assert(k < MAX_NUM_INDEX);
    }

    r.depth = k;
}

/* Add solution and permutation of solutions with prior solutions */
static void add_solution_with_permuations(std::vector<solution_t> &perm_sarray, solution_t &new_s)
{
    int i;
    size_t old_num_solutions = perm_sarray.size();

    perm_sarray.push_back(new_s);

    for (i = 0; i < old_num_solutions; i++) {
        solution_t permuted;

        xor_solutions(perm_sarray[i], new_s, permuted);
        perm_sarray.push_back(permuted);
    }
}

/* For a given sets of solutions and a new solution, returns the combinations */
static void get_solution_permuations(std::vector<solution_t> &perm_sarray,
        solution_t &new_s, std::vector<solution_t> &output_sarray)
{
    int i;
    size_t num_solutions = perm_sarray.size();

    output_sarray.push_back(new_s);

    for (i = 0; i < num_solutions; i++) {
        solution_t permuted;

        xor_solutions(perm_sarray[i], new_s, permuted);
        output_sarray.push_back(permuted);
    }
}
/* Returns the partition number according to the key and hypothesis */
static bool get_partition_num(uintptr_t key, const solution_t &s)
{
    int i;
    bool partition = 0;
    
    assert(s.depth > 0);

    for (i = 0; i < s.depth; i++) {
        partition = partition ^ ((key >> s.indexes[i]) & 0x1);
    }

    return partition;
}

/* Returns the full partition number according to the key and a set of hypothesis */
static int get_full_partitions_num(uintptr_t key, std::vector<solution_t> &solutions)
{
    int partition = 0;

    /* Check if possible overflow */
    assert(solutions.size() < sizeof(int) * 8);

    for (int i = 0; i < solutions.size(); i++) {
        partition |= (int)(get_partition_num(key, solutions[i])) << i;
    }

    return partition;
}

/* Finds the minimum bit in a set of solutions */
static int find_min_bit(std::vector<solution_t> &solutions)
{
    /* Bits in solutions are sorted ascendingly */
    int ret = INT_MAX;

    assert(solutions.size() > 0);

    for (int i = 0; i < solutions.size(); i++) {
        assert(solutions[i].depth > 0);
        ret = std::min(ret, solutions[i].indexes[0]);
    }

    return ret;
}

/* Checks if a solution is valid for all the keys */
static bool is_solution_correct(std::vector< std::pair <uintptr_t, uintptr_t> > &keys, const solution_t &s)
{
    size_t i;
    
    assert(s.depth >= 1);

    for (i = 0; i < keys.size(); i++) {

        bool partition1 = get_partition_num(keys[i].first, s);
        bool partition2 = get_partition_num(keys[i].second, s);

        if (partition1 != partition2)
            return false;
    }

    return true;
}

/* 
 * Find all the hash functions with which all the keys fit (i.e. lie on same
 * partition).
 * Return the number of solutions found.
 */
static int find_new_solutions(std::vector< std::pair <uintptr_t, uintptr_t> > &keys, int min_bit, int max_bit, 
        std::vector<solution_t> &solutions)
{
    int depth;
    int solutions_found = 0;
    bool ret;

    assert(solutions.size() == 0);

    for (depth = 1; depth <= max_bit - min_bit + 1; depth++) {
    
        int isFirst = 1;

        while (1) {
            solution_t s;
            s.depth = depth;

            if (permute_hypothesis(s.indexes, s.depth, min_bit, max_bit, isFirst) < 0)
                break;
            
            ret = is_solution_correct(keys, s);
            if (ret) {
                solutions.push_back(s);
                solutions_found++;
            }
            
            isFirst = 0;
        }
    }

    return solutions_found;
}

/* 
 * Checks if all keys fit with all the solutions.
 * Returns the number of solutions removed.
 */
static int remove_incorrect_solutions(hash_context_t *ctx)
{
	int i;
    int solutions_removed = 0;
    std::vector<solution_t>::iterator s;

    for (s = ctx->solutions.begin(); s != ctx->solutions.end();) {

        if (!is_solution_correct(ctx->keys, *s)) {
            solutions_removed++;
            s = ctx->solutions.erase(s);
        } else {
            ++s;
        }
    }

    return solutions_removed;
}

/* 
 * Checks if the number of solutions seems correct or are they more than 
 * they need to be.
 */
static bool are_unique_solutions_found(size_t num_solutions, int min_bit, int max_bit)
{
    /* 
    * Max number of solutions can be only be (max_bit - min_bit - 1)
    * Since n solutions give 2^n partitions
    */
    return (num_solutions <= (max_bit - min_bit + 1));
}

/* 
 * Finds solutions based on keys found.
 * Returns true if found all solutions.
 * Returns false if not sure if all solutions found.
 */
static bool try_find_all_solutions(hash_context_t *ctx)
{
    /* Do we have atleast a pair of keys to compare */
    if (ctx->keys.size() != 0) {

        /* Some solutions already exist? */
        if (ctx->solutions.size() > 0) {
            if (remove_incorrect_solutions(ctx) == 0) {

                hash_reduce(ctx);

                /* 
                 * Max number of solutions can be only be (max_bit - min_bit - 1)
                 * Since n solutions give 2^n partitions
                 */
                if (are_unique_solutions_found(ctx->solutions.size(), ctx->max_bit,
                            ctx->min_bit))
                    return 1;
            }
        } else {

            find_new_solutions(ctx->keys, ctx->min_bit, ctx->max_bit, 
                    ctx->solutions);
        }  
    }

    return 0;
}
 
/* Find highest bit set in mask <= ceiling bit */
static int find_highest_bit(uintptr_t mask, int ceiling)
{
    int start = std::min(ceiling, MAX_NUM_INDEX - 1);
    for (int i = start; i >= 0; i--) {
        if (mask & (1ULL << i))
            return i;
    }

    return -1;
}

/* Called when is it confirmed that the pair of address lie in same partition */
static void hash_confirm_pair(hash_context_t *ctx, uintptr_t phy_addr1, uintptr_t phy_addr2)
{
    std::pair <uintptr_t, uintptr_t> key (phy_addr1, phy_addr2);

    ctx->keys.push_back(key);
}

static void eliminate_duplicate_solutions(std::vector<solution_t> &solutions)
{
    std::vector<solution_t> perm_sarray;
    std::vector<solution_t>::iterator s;

    for (s = solutions.begin(); s != solutions.end();) {

        bool is_unique = true;
        
        for (size_t i = 0; i < perm_sarray.size(); i++) {
            if (are_solutions_same(*s, perm_sarray[i])) {
                is_unique = false;
                break;
            }   
        }

        if (is_unique) {
            try {
                add_solution_with_permuations(perm_sarray, *s);
            } catch(...) {
                /* Out of memory - Permutation grows exponentially */
                fprintf(stderr, "Out of memory exception\n");
                return;
            }
            s++;
        } else {
            s = solutions.erase(s);
        }
    }

}

static bool solution_sort_ascending_cb(solution_t a, solution_t b)
{
    assert(a.depth >= 1);
    assert(b.depth >= 1);

    return a.indexes[0] < b.indexes[0];
}


static bool solution_sort_descending_cb(solution_t a, solution_t b)
{
    assert(a.depth >= 1);
    assert(b.depth >= 1);

    return a.indexes[0] > b.indexes[0];
}

/* Given a set of unique solutions, find those permutations that 
 * give the highest lowest starting indexes.
 */
static void sort_solutions(std::vector<solution_t> &solutions)
{
    std::vector<solution_t> perm_sarray;
    std::vector<solution_t>::iterator s, v, v_max;
    std::vector<solution_t> sorted;
    int max_index;

    /* First, sort solutions by first index */
    std::sort (solutions.begin(), solutions.end(), solution_sort_ascending_cb);

    for (s = solutions.begin(); s != solutions.end(); s++) {

        std::vector<solution_t> variations;

        /* Get all variations */
        get_solution_permuations(perm_sarray, *s, variations);

        /* Find the one with highest lowest index (Don't need to sort - Wasteful) */
        for (v = variations.begin(), max_index = INT_MIN; 
                v != variations.end(); v++) {

            assert(v->depth >= 1);

            if (v->indexes[0] > max_index) {
                max_index = v->indexes[0];
                v_max = v;
            }
        }

        sorted.push_back(*v_max);

        /* Collect all the variations */
        add_solution_with_permuations(perm_sarray, *s);
    }

    solutions = sorted;

    /* Solutions might get unsorted */
    std::sort (solutions.begin(), solutions.end(), solution_sort_ascending_cb);
}

/* 
 * Finds solutions unique to solutions1 (not present in solutions2) 
 * and returns them in out 
 */
static void get_unique_solutions(std::vector<solution_t> &solutions1,
        std::vector<solution_t> &solutions2, std::vector<solution_t> &out)
{
    std::vector<solution_t> perm_sarray;
    std::vector<solution_t>::iterator s1, s2;

    out.clear();

    /* Get all permutations of solutions in solution2 */
    for (s2 = solutions2.begin(); s2 != solutions2.end(); s2++) {

        try {
            add_solution_with_permuations(perm_sarray, *s2);
        } catch(...) {
            /* Out of memory - Permutation grows exponentially */
            fprintf(stderr, "Out of memory exception\n");
            return;
        }
    }

    /* We want to keep solutions with higher bits */
    std::sort (solutions1.begin(), solutions1.end(), solution_sort_descending_cb);

    /* A solution in solution1 is unique if it is not found in perm_sarray */
    for (s1 = solutions1.begin(); s1 != solutions1.end(); s1++) {

        bool is_unique = true;
        for (s2 = perm_sarray.begin(); s2 != perm_sarray.end(); s2++) {
            if (are_solutions_same(*s1, *s2)) {
                is_unique = false;
                break;
            }
        }

        if (!is_unique)
            continue;

        out.push_back(*s1);

        /* Collect all the variations */
        add_solution_with_permuations(perm_sarray, *s1);
    }
}

/* Eliminate equivalent solutions to get only unique solutions */
static void hash_reduce(hash_context_t *ctx)
{
    if (ctx->solutions.size() == 0)
        return;

    eliminate_duplicate_solutions(ctx->solutions);
}
/* 
 * Initializes the hash context. 
 * Min bit is the minimum bit to participate in the hash functions.
 * Max bit is the maximum bit to participate in the hash functions.
 * Start and End address are the range of address that can be tested.
 */
hash_context_t *hash_init(int min_bit, int max_bit,
        void *start_addr, void *end_addr)
{
    size_t length;
    hash_context_t *ctx = NULL;

    if (max_bit <= min_bit)
        return NULL;
    
    if (max_bit >= MAX_NUM_INDEX)
        return NULL;

    if (min_bit < 0)
        return NULL;

    if ((uintptr_t)end_addr <= (uintptr_t)start_addr)
        return NULL;

    /* Do we have sufficient address space to find all the bits? */
    length = (uintptr_t)end_addr - (uintptr_t)start_addr;
    if (length < (1ULL << max_bit))
        return NULL;

    ctx = new hash_context_t();
    
    ctx->min_bit = min_bit;
    ctx->max_bit = max_bit;
    
    ctx->start_addr = (uintptr_t)start_addr;
    ctx->end_addr = (uintptr_t)end_addr;

    return ctx;
}

/*
 * Give a set of solutions, try seeing if a new bit can 
 * be added into the sets of solutions.
 */
static void try_accomodate_new_bit(hash_context_t *ctx, int new_bit, void *arg,
        void *(*find_next_partition_pair)(void *addr1, void *start_addr, 
        void *end_addr, size_t offset, void *arg))
{
    std::vector< std::vector<solution_t> > all_new_solutions;
    std::vector<solution_t> new_solutions;
    int ret;
    int num_solutions = ctx->solutions.size();
    uintptr_t base_addr, test_addr;

    assert(num_solutions < 8 * sizeof(uint64_t));

    /* Permute all new solutions */
    for (uint64_t i = 0; i <= (1ULL << num_solutions) - 1; i++) {
        new_solutions = ctx->solutions;
        for (int j = 0; j < 8 * sizeof(uint64_t); j++) {
            if (i & (1ULL << j)) {
                ret = insert_bit_in_solutions(new_solutions[j], new_bit);
                assert(ret >= 0);
            }
        }
        all_new_solutions.push_back(new_solutions);
    }

    /* Try eliminating all but one of the new solutions */
    uintptr_t end_addr = ctx->start_addr + (1ULL << (new_bit + 1));
    end_addr = std::min(ctx->end_addr, end_addr);
    base_addr = ctx->start_addr;
    
    size_t offset = 1ULL << ctx->min_bit;

    for (test_addr = ctx->start_addr + (1ULL << new_bit); 
            test_addr <= end_addr; 
            test_addr += offset) {
        
        void *addr = find_next_partition_pair((void *)base_addr, 
                (void *)test_addr, (void *)end_addr, offset, arg);
        if (addr) {
            
            std::vector< std::vector<solution_t> >::iterator s;

            test_addr = (uintptr_t)addr;
            hash_confirm_pair(ctx, base_addr, test_addr);
       
            for (s = all_new_solutions.begin(); s != all_new_solutions.end();) {
                
                bool is_correct = true;

                for (int i = 0; i < s->size(); i++) {
                    if (!is_solution_correct(ctx->keys, (*s)[i])) {
                        is_correct = false;
                        break;
                    }
                }

                if (!is_correct) {
                    s = all_new_solutions.erase(s);
                } else {
                    s++;
                }
            }
        } else {
            break;
        }

        if (all_new_solutions.size() <= 1)
            break;
    }

    if (all_new_solutions.size() == 0) {
        fprintf(stderr, "Something went wrong\n");
        return;
    }

    ctx->solutions = all_new_solutions[0];
}

/* 
 * Runs till a solution is found.
 * Takes a callback function as argument:
 *   Given an address 'addr1', find another address between [start_addr, end_addr]
 *   (testing at offset of 'offset') that lies in same partition as 'addr1' and
 *   return it. Return NULL if no such pair found.
 *
 * Returns 0 if solution found.
 * Returns < 0 if no solutions found.
 */
int hash_find_solutions(hash_context_t *ctx, void *arg,
    void *(*find_next_partition_pair)(void *addr1, void *start_addr, 
        void *end_addr, size_t offset, void *arg))
{
    /* 
     * Works in two steps:
     * 1) Find a base solution - 
     *   This is done via brute force. (XXX: Any faster method)
     *   To make it fast, only consider half of the total bits to consider.
     * 2) Using base solution, find out the role of each leftover bit seperately
     * Using only half of the bits in finding base solution exponentially
     * speeds up the process since each extra bit makes the brute force solution
     * 2x slower whereas second step is O(1) for each leftover bit.
     */
    int end_bit = ((ctx->max_bit + ctx->min_bit) + 1) / 2;
    int highest_bit = find_highest_bit(ctx->start_addr, ctx->max_bit);
    end_bit = std::max(end_bit, highest_bit + 1);
    uintptr_t base_addr, test_addr;

    uintptr_t end_addr = (1ULL << (end_bit + 1)) - 1;
    end_addr = std::min(ctx->end_addr, end_addr);
    std::vector<solution_t> new_solutions;

    /* Step 1 */
    printf("Finding base solutions\n");
    base_addr = ctx->start_addr;
    test_addr = ctx->start_addr;

    size_t count;
    size_t max_count = (end_addr - ctx->start_addr) / (1ULL << ctx->min_bit) + 1;
    size_t offset = 1ULL << ctx->min_bit;

    for (test_addr = ctx->start_addr; test_addr <= end_addr; 
            test_addr += offset) {

        void *addr = find_next_partition_pair((void *)base_addr, (void *)test_addr,
                (void *)end_addr, offset, arg);
        if (addr) {
            hash_confirm_pair(ctx, base_addr, (uintptr_t)addr);
            test_addr = (uintptr_t)addr;
        } else {
            break;
        }

        count = (test_addr - ctx->start_addr)/offset;
        /* Print progress */
        printf("Done:%.1f%%\r", (float)(count * 100)/(float)(max_count));
        fflush(stdout);
    }
    printf("\n");

    if (ctx->keys.size() == 0) {
        fprintf(stderr, "Base solution couldn't be found as no pair found\n");
        return -1;
    }

    if (find_new_solutions(ctx->keys, ctx->min_bit, end_bit, ctx->solutions) == 0) {
        fprintf(stderr, "Base solution couldn't be found\n");
        return -1;
    }

    hash_reduce(ctx);

    /* XXX: Remove this check and try eliminate frivilous solutions */
    if (!are_unique_solutions_found(ctx->solutions.size(), ctx->min_bit, end_bit)) {
        fprintf(stderr, "Too many base solutions\n");
        return -1;
    }

    /* Step 2 */
    count = 0;
    max_count = (ctx->max_bit - end_bit);
    printf("Finding overall solutions\n");
    for (int i = end_bit + 1; i <= ctx->max_bit; count++, i++) {
        try_accomodate_new_bit(ctx, i, arg, find_next_partition_pair);
        /* Print progress */
        printf("Done:%.1f%%\r", (float)(count * 100)/(float)(max_count));
        fflush(stdout);
    }
    printf("\n");

    return 0;
}

typedef struct cb_arg {
    bool (*check_partition_pair)(void *addr1, void *addr2, void *arg);
    void *arg;
} cb_arg_t;

static void *local_find_next_partition_pair(void *addr1, void *start_addr, 
        void *end_addr, size_t offset, void *arg)
{
    cb_arg_t *data = (cb_arg_t *)arg;
    uintptr_t uaddr2;

    for (uaddr2 = (uintptr_t)start_addr; uaddr2 <= (uintptr_t)end_addr;
            uaddr2 += offset) {
        if (data->check_partition_pair(addr1, (void *)uaddr2, data->arg))
            return (void *)uaddr2;
    }

    return NULL;
}

/* 
 * Runs till a solution is found.
 * Takes a callback function as argument:
 *   Given a pair of addresses 'addr1' and 'addr2', checks if they lie on same
 *   partition.
 *
 * Returns 0 if solution found.
 * Returns < 0 if no solutions found.
 */
int hash_find_solutions2(hash_context_t *ctx, void *arg,
    bool (*check_partition_pair)(void *addr1, void *addr2, void *arg))
{
    cb_arg_t data;

    data.arg = arg;
    data.check_partition_pair = check_partition_pair;

    return hash_find_solutions(ctx, &data, local_find_next_partition_pair);
}

void hash_print_solutions(hash_context_t *ctx)
{
    for (size_t i = 0; i < ctx->solutions.size(); i++)
        print_solution(ctx->solutions[i]);
}

void hash_sort_solutions(hash_context_t *ctx)
{
    sort_solutions(ctx->solutions);
}

/* Finds common solutions and print them. Also remove the common solutions */
hash_context_t *hash_get_common_solutions(hash_context_t *ctx1, hash_context_t *ctx2)
{
    hash_context_t *hctx_new;
    std::vector<solution_t> perm_sarray1;
    std::vector<solution_t> perm_sarray2;
    std::vector<solution_t> common, unique;
    std::vector<solution_t>::iterator s1, s2;

    hash_reduce(ctx1);
    hash_reduce(ctx2);

    for (s1 = ctx1->solutions.begin(); s1 != ctx1->solutions.end(); s1++) {

        try {
            add_solution_with_permuations(perm_sarray1, *s1);
        } catch(...) {
            /* Out of memory - Permutation grows exponentially */
            fprintf(stderr, "Out of memory exception while hash_reduce\n");
            return NULL;
        }
    }

    for (s2 = ctx2->solutions.begin(); s2 != ctx2->solutions.end(); s2++) {

        try {
            add_solution_with_permuations(perm_sarray2, *s2);
        } catch(...) {
            /* Out of memory - Permutation grows exponentially */
            fprintf(stderr, "Out of memory exception while hash_reduce\n");
            return NULL;
        }
    }

    for (s1 = perm_sarray1.begin(); s1 != perm_sarray1.end(); ) {
        
        bool is_unique = true;
        for (s2 = perm_sarray2.begin(); s2 != perm_sarray2.end(); ) {
            if (are_solutions_same(*s1, *s2)) {
                is_unique = false;
                common.push_back(*s1);
                s2 = perm_sarray2.erase(s2);
                break;
            } else {
                s2++;
            }
        }

        if (is_unique) {
            s1++;
        } else {
            s1 = perm_sarray1.erase(s1);
        }
    }

    eliminate_duplicate_solutions(common);

    get_unique_solutions(ctx1->solutions, common, unique);
    ctx1->solutions = unique;

    get_unique_solutions(ctx2->solutions, common, unique);
    ctx2->solutions = unique;

    /* Sort before printing */
    sort_solutions(common);
    sort_solutions(ctx1->solutions);
    sort_solutions(ctx2->solutions);

    printf("Number of common solutions found: %zd\n", common.size());
    for (size_t i = 0; i < common.size(); i++)
        print_solution(common[i]);

    hctx_new = new(hash_context_t);
    hctx_new->solutions = common;
    hctx_new->min_bit = std::max(ctx1->min_bit, ctx2->min_bit);
    hctx_new->max_bit = std::min(ctx1->max_bit, ctx2->max_bit);

    hctx_new->start_addr = std::max(ctx1->start_addr, ctx2->start_addr);
    hctx_new->end_addr = std::min(ctx1->end_addr, ctx2->end_addr);

    return hctx_new;
}

/* 
 * Returns a new address (between (start_addr, end_addr]) that matches solutions
 * and addr
 */
static void *get_next_addr(std::vector<solution_t> &solutions, int partition, 
        void *_start_addr, void *_end_addr)
{
    uintptr_t start_addr = (uintptr_t)_start_addr;
    uintptr_t end_addr = (uintptr_t)_end_addr;
    uintptr_t i;
    int min_bit = find_min_bit(solutions);
    size_t min_offset = 1ULL << min_bit;

    /* Round up start_addr */
    start_addr++;
    start_addr = (start_addr + min_offset - 1) & ~(min_offset - 1);

    /* Can speedup by finding mininum offset in the solutions */
    for (i = start_addr; i <= end_addr; i += min_offset) {
       
        if (partition == get_full_partitions_num(i, solutions))
            break;
    }

    if (i > end_addr)
        return NULL;

    return (void *)i;
}

/* 
 * Returns a new address (between (start_addr, end_addr]) that matches solutions
 * thr pair of ctx and partitions numbers
 */

void *hash_get_next_addr(std::vector<hash_context_t *> ctx, 
        std::vector<int> partition, void *start_addr, void *end_addr)
{
    int min_bit = INT_MAX;
    int min_index;
    void *addr;
    bool found = false;
    void *prev_addr = start_addr;

    assert(partition.size() == ctx.size());
    assert(ctx.size() >= 1);

    for (int i = 0; i < ctx.size(); i++) {
        int c_min_bit = find_min_bit(ctx[i]->solutions);
        if (c_min_bit < min_bit) {
            min_index = i;
            min_bit = c_min_bit;
        }
    }

    while (!found) {
        addr = get_next_addr(ctx[min_index]->solutions, 
                partition[min_index], prev_addr, end_addr);

        if (!addr)
            return NULL;

        found = true;

        for (int i = 0; i < ctx.size(); i++) {
            if (i == min_index)
                continue;
            if (get_full_partitions_num((uintptr_t)addr, ctx[i]->solutions) !=
                    partition[i]) {
                found = false;
                break;
            }
        }

        prev_addr = addr;
    }
    
    if (addr > end_addr)
        return NULL;

    return addr;
}


/* Checks if both address lie on same partition according to set of solutions of ctx */
bool hash_is_same_partition(hash_context_t *ctx, void *addr1, void *addr2)
{
    return get_full_partitions_num((uintptr_t)addr1, ctx->solutions) ==
        get_full_partitions_num((uintptr_t)addr1, ctx->solutions);
}

void hash_del(hash_context_t *ctx)
{
    delete ctx;
}

/* TODO: Below code is just to make testing faster. Remove. For GTX 1070*/
#if 0
hash_context_t *hash_get_dram(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {13, 15, 20, 24, 26, 29, 32}, .depth = 7};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {15, 16, 21, 22, 23, 25, 26, 28, 29}, .depth = 9};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {16, 19, 23, 27, 30}, .depth = 5};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {17, 20, 22, 23, 24, 27, 28, 29, 31}, .depth = 9};
        ctx->solutions.push_back(s);
    }

    return ctx;
}

hash_context_t *hash_get_cache(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {7, 8, 16, 17, 23, 26, 31}, .depth = 7};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {8, 10, 12, 16, 17, 21, 24, 25, 26, 27}, .depth = 10};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {9, 10, 18, 25, 29, 30, 31}, .depth = 7};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {13, 14, 20, 23, 28, 29, 30}, .depth = 7};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {14, 15, 17, 20, 21, 23, 24, 28, 31}, .depth = 9};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {15, 16, 19, 20, 23, 24, 25, 26, 28, 29, 30, 32}, .depth = 12};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 30}, .depth = 11};
        ctx->solutions.push_back(s);
    }

    return ctx;
}

hash_context_t *hash_get_common(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {10, 12, 16, 20, 23, 26, 29, 30}, .depth = 8};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30}, .depth = 11};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {12, 13, 18, 19, 22, 25, 26, 27, 30, 31}, .depth = 10};
        ctx->solutions.push_back(s);
    }

    return ctx;
}
#endif

/* TODO: Below code is just to make testing faster. Remove. For V100*/
#if 0
hash_context_t *hash_get_dram(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {16, 19, 20, 23, 27, 29, 31, 33}, .depth = 8};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {17, 18, 23, 24, 25, 27, 28, 30, 31}, .depth = 9};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {18, 21, 25, 29, 32}, .depth = 5};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {19, 20, 22, 24, 25, 26, 29, 30, 31, 33}, .depth = 10};
        ctx->solutions.push_back(s);
    }

    return ctx;
}

hash_context_t *hash_get_cache(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {7, 15, 21, 23, 24, 25, 28, 32}, .depth = 8};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {8, 9, 15, 19, 21, 23, 24, 25, 28, 29, 31}, .depth = 11};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {9, 15, 19, 21, 22, 24, 25, 26, 27, 30, 31, 32, 33}, .depth = 13};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {14, 19, 20, 24, 25, 27, 28, 29, 30, 31, 32, 33}, .depth = 12};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {16, 19, 21, 22, 25, 26, 28, 30, 32}, .depth = 9};
        ctx->solutions.push_back(s);
    }

    return ctx;
}

hash_context_t *hash_get_common(void)
{
    hash_context_t *ctx = new(hash_context_t);
    {
        solution_t s = {.indexes = {10, 13, 17, 19, 24, 25, 26, 29, 30, 32, 33}, .depth = 11};
        ctx->solutions.push_back(s);
    }  
    {
        solution_t s = {.indexes = {11, 13, 15, 23, 24, 26, 27, 29, 30, 31}, .depth = 10};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {12, 15, 16, 18, 20, 21, 23, 26, 28, 29, 30}, .depth = 11};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {13, 19, 20, 22, 25, 27, 28, 29}, .depth = 8};
        ctx->solutions.push_back(s);
    }
    {
        solution_t s = {.indexes = {15, 18, 22, 23, 26, 29, 30, 31, 32, 33}, .depth = 10};
        ctx->solutions.push_back(s);
    }

    return ctx;
}
#endif
