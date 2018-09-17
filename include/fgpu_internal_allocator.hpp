/* Data structures used by the custom allocator */
#ifndef __FGPU_INTERNAL_ALLOCATOR_HPP__
#define __FGPU_INTERNAL_ALLOCATOR_HPP__

/* Generalized queue implementation */
#include <assert.h>

/*
 * Generates a new structure of type Q_HEAD_TYPE representing the head
 * of a queue of elements of type Q_ELEM_TYPE.
 * 
 * Usage: Q_NEW_HEAD(Q_HEAD_TYPE, Q_ELEM_TYPE); //create the type <br>
           Q_HEAD_TYPE headName; //instantiate a head of the given type
 *
 *  Q_HEAD_TYPE: the type you wish the newly-generated structure to have.
 *
 *  Q_ELEM_TYPE: the type of elements stored in the queue.
 *  Q_ELEM_TYPE must be a structure.
 */
#define Q_NEW_HEAD(Q_HEAD_TYPE, Q_ELEM_TYPE)    \
    typedef struct Q_HEAD_TYPE {                \
        struct Q_ELEM_TYPE *head;               \
        struct Q_ELEM_TYPE *tail;               \
        size_t size;                            \
    } Q_HEAD_TYPE

/*
 * Instantiates a link within a structure, allowing that structure to
 * be collected into a queue created with Q_NEW_HEAD.
 *
 * Usage:
 *  typedef struct Q_ELEM_TYPE {
 *  Q_NEW_LINK(Q_ELEM_TYPE) LINK_NAME; //instantiate the link
 *  } Q_ELEM_TYPE;
 *
 * Q_ELEM_TYPE: the type of the structure containing the link
 */
#define Q_NEW_LINK(Q_ELEM_TYPE)         \
    struct {                            \
        struct Q_ELEM_TYPE *prev;       \
        struct Q_ELEM_TYPE *next;       \
    }

/*
 *  Initializes the head of a queue so that the queue head can be used
 *  properly.
 *  Q_HEAD Pointer: to queue head to initialize
 */
#define Q_INIT_HEAD(Q_HEAD)     \
    do {                        \
        (Q_HEAD)->head = NULL;  \
        (Q_HEAD)->tail = NULL;  \
        (Q_HEAD)->size  = 0;    \
    } while(0)

/*
 * Copies the head of a queue
 * Q_HEAD_DEST: Destination head
 * Q_HEAD_SRC: Source head
 */ 
#define Q_COPY_HEAD(Q_HEAD_DEST, Q_HEAD_SRC)    		\
    do {                       					        \
        (Q_HEAD_DEST)->head = (Q_HEAD_SRC)->head;		\
        (Q_HEAD_DEST)->tail = (Q_HEAD_SRC)->tail;		\
        (Q_HEAD_DEST)->size  = (Q_HEAD_SRC)->size;		\
    } while(0)

/*
 *  Initializes the link named LINK_NAME in an instance of the structure
 *  Q_ELEM.
 *
 *  Once initialized, the link can be used to organized elements in a queue.
 *
 *  Q_ELEM: Pointer to the structure instance containing the link
 *  LINK_NAME: The name of the link to initialize
 */
#define Q_INIT_ELEM(Q_ELEM, LINK_NAME)      \
    do {                                    \
        (Q_ELEM)->LINK_NAME.prev = NULL;    \
        (Q_ELEM)->LINK_NAME.next = NULL;    \
    } while (0)

/*
 *  Inserts the queue element pointed to by Q_ELEM at the front of the
 *  queue headed by the structure Q_HEAD.
 *
 *  The link identified by LINK_NAME will be used to organize the element and
 *  record its location in the queue.
 *
 *  Q_HEAD: Pointer to the head of the queue into which Q_ELEM will be
 *  inserted
 *  Q_ELEM: Pointer to the element to insert into the queue
 *  LINK_NAME: Name of the link used to organize the queue
 *
 */
#define Q_INSERT_FRONT(Q_HEAD, Q_ELEM, LINK_NAME)           \
    do {                                                    \
        (Q_HEAD)->size++;                                   \
                                                            \
        /* Not empty queue ? */                             \
        if ((Q_HEAD)->head != NULL) {                       \
            (Q_HEAD)->head->LINK_NAME.prev = (Q_ELEM);      \
            (Q_ELEM)->LINK_NAME.next = (Q_HEAD)->head;      \
            (Q_ELEM)->LINK_NAME.prev = NULL;                \
            (Q_HEAD)->head = (Q_ELEM);                      \
        } else {                                            \
            (Q_HEAD)->head = (Q_ELEM);                      \
            (Q_HEAD)->tail = (Q_ELEM);                      \
            (Q_ELEM)->LINK_NAME.prev = NULL;                \
            (Q_ELEM)->LINK_NAME.next = NULL;                \
        }                                                   \
    } while (0)

/*
 *  nserts the queue element pointed to by Q_ELEM at the end of the
 *  queue headed by the structure pointed to by Q_HEAD.
 *
 *  The link identified by LINK_NAME will be used to organize the element and
 *  record its location in the queue.
 *
 *  Q_HEAD: Pointer to the head of the queue into which Q_ELEM will be
 *  inserted
 *  Q_ELEM: Pointer to the element to insert into the queue
 *  LINK_NAME: Name of the link used to organize the queue
 */
#define Q_INSERT_TAIL(Q_HEAD, Q_ELEM, LINK_NAME)            \
    do {                                                    \
        (Q_HEAD)->size++;                                   \
                                                            \
        /* Not empty queue ? */                             \
        if ((Q_HEAD)->tail != NULL) {                       \
            (Q_HEAD)->tail->LINK_NAME.next = (Q_ELEM);      \
            (Q_ELEM)->LINK_NAME.prev = (Q_HEAD)->tail;      \
            (Q_ELEM)->LINK_NAME.next = NULL;                \
            (Q_HEAD)->tail = (Q_ELEM);                      \
        } else {                                            \
            (Q_HEAD)->head = (Q_ELEM);                      \
            (Q_HEAD)->tail = (Q_ELEM);                      \
            (Q_ELEM)->LINK_NAME.prev = NULL;                \
            (Q_ELEM)->LINK_NAME.next = NULL;                \
        }                                                   \
    } while (0)

/*
 *  Returns a pointer to the first element in the queue, or NULL
 *  (memory address 0) if the queue is empty.
 *
 *  Q_HEAD: Pointer to the head of the queue
 *  Returns pointer to the first element in the queue, or NULL if the queue
 *  is empty
 */
#define Q_GET_FRONT(Q_HEAD) \
    (Q_HEAD)->head

/*
 *  Returns a pointer to the last element in the queue, or NULL
 *  (memory address 0) if the queue is empty.
 *
 *  Q_HEAD: Pointer to the head of the queue
 *  Returns pointer to the last element in the queue, or NULL if the queue
 *  is empty
 */
#define Q_GET_TAIL(Q_HEAD) \
    (Q_HEAD)->tail

/*
 *  Returns a pointer to the next element in the queue, as linked to by
 *  the link specified with LINK_NAME.
 *
 *  If Q_ELEM is not in a queue or is the last element in the queue,
 *  Q_GET_NEXT should return NULL.
 *
 *  Q_ELEM: Pointer to the queue element before the desired element
 *  LINK_NAME: Name of the link organizing the queue
 *
 *  Returns the element after Q_ELEM, or NULL if there is no next element
 */
#define Q_GET_NEXT(Q_ELEM, LINK_NAME) \
    (Q_ELEM)->LINK_NAME.next

/*
 *  Returns a pointer to the previous element in the queue, as linked to
 *  by the link specified with LINK_NAME.
 *
 *  If Q_ELEM is not in a queue or is the first element in the queue,
 *  Q_GET_NEXT should return NULL.
 *
 *  Q_ELEM: Pointer to the queue element after the desired element
 *  LINK_NAME: Name of the link organizing the queue
 *
 *  Returns the element before Q_ELEM, or NULL if there is no next element
 **/
#define Q_GET_PREV(Q_ELEM, LINK_NAME) \
    (Q_ELEM)->LINK_NAME.prev

/*
 *  Inserts the queue element Q_TOINSERT after the element Q_INQ
 *  in the queue.
 *
 *  Inserts an element into a queue after a given element. If the given
 *  element is the last element, Q_HEAD should be updated appropriately
 *  (so that Q_TOINSERT becomes the tail element)
 *
 *  Q_HEAD: head of the queue into which Q_TOINSERT will be inserted
 *  Q_INQ:  Element already in the queue
 *  Q_TOINSERT: Element to insert into queue
 *  LINK_NAME:  Name of link field used to organize the queue
 */
#define Q_INSERT_AFTER(Q_HEAD,Q_INQ,Q_TOINSERT,LINK_NAME)               \
    do {                                                                \
        (Q_HEAD)->size++;                                               \
                                                                        \
        (Q_TOINSERT)->LINK_NAME.next = (Q_INQ)->LINK_NAME.next;         \
        (Q_TOINSERT)->LINK_NAME.prev = (Q_INQ);                         \
        if ((Q_INQ)->LINK_NAME.next != NULL)                            \
            (Q_INQ)->LINK_NAME.next->LINK_NAME.prev = (Q_TOINSERT);     \
        (Q_INQ)->LINK_NAME.next = (Q_TOINSERT);                         \
                                                                        \
        /* Update tail */                                               \
        if ((Q_INQ) == (Q_HEAD)->tail)                                  \
            (Q_HEAD)->tail = (Q_TOINSERT);                              \
    } while (0)

/*
 *  Inserts the queue element Q_TOINSERT before the element Q_INQ
 *  in the queue.
 *
 *  Inserts an element into a queue before a given element. If the given
 *  element is the first element, Q_HEAD should be updated appropriately
 *  (so that Q_TOINSERT becomes the front element)
 *
 *  Q_HEAD: head of the queue into which Q_TOINSERT will be inserted
 *  Q_INQ:  Element already in the queue
 *  Q_TOINSERT: Element to insert into queue
 *  LINK_NAME:  Name of link field used to organize the queue
 */
#define Q_INSERT_BEFORE(Q_HEAD,Q_INQ,Q_TOINSERT,LINK_NAME)              \
    do {                                                                \
        (Q_HEAD)->size++;                                               \
                                                                        \
        (Q_TOINSERT)->LINK_NAME.prev = (Q_INQ)->LINK_NAME.prev;         \
        (Q_TOINSERT)->LINK_NAME.next = (Q_INQ);                         \
        if ((Q_INQ)->LINK_NAME.prev != NULL)                            \
            (Q_INQ)->LINK_NAME.prev->LINK_NAME.next = (Q_TOINSERT);     \
        (Q_INQ)->LINK_NAME.prev = (Q_TOINSERT);                         \
                                                                        \
        /* Update head */                                               \
        if ((Q_INQ) == (Q_HEAD)->head)                                  \
            (Q_HEAD)->head = (Q_TOINSERT);                              \
    } while (0)

/*
 *  Detaches the element Q_ELEM from the queue organized by LINK_NAME,
 *  and returns a pointer to the element.
 *
 *  If Q_HEAD does not use the link named LINK_NAME to organize its elements or
 *  if Q_ELEM is not a member of Q_HEAD's queue, the behavior of this macro
 *  is undefined.
 *
 *  Q_HEAD: Pointer to the head of the queue containing Q_ELEM. If
 *  Q_REMOVE removes the first, last, or only element in the queue,
 *  Q_HEAD should be updated appropriately.
 *  Q_ELEM: Pointer to the element to remove from the queue headed by
 *  Q_HEAD.
 *  LINK_NAME: The name of the link used to organize Q_HEAD's queue
 */
#define Q_REMOVE(Q_HEAD,Q_ELEM,LINK_NAME)                               \
    do {                                                                \
        if ((Q_HEAD)->size > 0) {                                       \
            (Q_HEAD)->size--;                                           \
                                                                        \
            if ((Q_ELEM)->LINK_NAME.next != NULL)                       \
                (Q_ELEM)->LINK_NAME.next->LINK_NAME.prev =              \
                    (Q_ELEM)->LINK_NAME.prev;                           \
                                                                        \
           if ((Q_ELEM)->LINK_NAME.prev != NULL)                        \
                (Q_ELEM)->LINK_NAME.prev->LINK_NAME.next =              \
                    (Q_ELEM)->LINK_NAME.next;                           \
                                                                        \
            /* Update head */                                           \
            if ((Q_ELEM) == (Q_HEAD)->head)                             \
                (Q_HEAD)->head = (Q_ELEM)->LINK_NAME.next;              \
                                                                        \
            /* Update tail */                                           \
            if ((Q_ELEM) == (Q_HEAD)->tail)                             \
                (Q_HEAD)->tail = (Q_ELEM)->LINK_NAME.prev;              \
                                                                        \
            (Q_ELEM)->LINK_NAME.next = NULL;                            \
            (Q_ELEM)->LINK_NAME.prev = NULL;                            \
        } \
    } while (0)

/*
 *  Asserts that the element is not in a queue currently
 *  Q_ELEM: The element
 *  LINK_NAME: The link name for insertion in a queue
 */
#define Q_CHECK_REMOVED(Q_ELEM,LINK_NAME)				\
	do {								\
		assert((Q_ELEM)->LINK_NAME.next == NULL);		\
		assert((Q_ELEM)->LINK_NAME.prev == NULL);		\
	} while (0)

/*
 *  Constructs an iterator block (like a for block) that operates
 *  on each element in Q_HEAD, in order.
 *
 *  Q_FOREACH constructs the head of a block of code that will iterate through
 *  each element in the queue headed by Q_HEAD. Each time through the loop,
 *  the variable named by CURRENT_ELEM will be set to point to a subsequent
 *  element in the queue.
 *
 *  Usage:
 *  Q_FOREACH(CURRENT_ELEM,Q_HEAD,LINK_NAME)
 *  {
 *  ... operate on the variable CURRENT_ELEM ...
 *  }
 *
 *  If LINK_NAME is not used to organize the queue headed by Q_HEAD, then
 *  the behavior of this macro is undefined.
 *
 *  CURRENT_ELEM: name of the variable to use for iteration. On each
 *  loop through the Q_FOREACH block, CURRENT_ELEM will point to the
 *  current element in the queue. CURRENT_ELEM should be an already-
 *  defined variable name, and its type should be a pointer to
 *  the type of data organized by Q_HEAD
 *  Q_HEAD: Pointer to the head of the queue to iterate through
 *  LINK_NAME: The name of the link used to organize the queue headed
 *  by Q_HEAD.
 */

#define Q_FOREACH(CURRENT_ELEM,Q_HEAD,LINK_NAME)                    \
    for ((CURRENT_ELEM) = (Q_HEAD)->head; (CURRENT_ELEM) != NULL;   \
         (CURRENT_ELEM) = (CURRENT_ELEM)->LINK_NAME.next)


/*
 *  Constructs an iterator block (like a for block) that operates
 *   on each element in Q_HEAD, in order.
 *
 *  Q_FOREACH_WITH_MODIFY constructs the head that will iterate through
 *  each element in the queue headed by Q_HEAD. Each time through the loop,
 *  the variable named by CURRENT_ELEM will be set to point to a subsequent
 *  element in the queue.
 *
 *  Usage:
 *  Q_FOREACH_WITH_MODIFY(CURRENT_ELEM,NEXT_ELEM, Q_HEAD,LINK_NAME)
 *  {
 *  ... operate on the variable CURRENT_ELEM ...
 *  }
 *
 *  If LINK_NAME is not used to organize the queue headed by Q_HEAD, then
 *  the behavior of this macro is undefined.
 *
 *  CURRENT_ELEM: name of the variable to use for iteration. On each
 *  loop through the Q_FOREACH block, CURRENT_ELEM will point to the
 *  current element in the queue. CURRENT_ELEM should be an already-
 *  defined variable name, and its type should be a pointer to
 *  the type of data organized by Q_HEAD
 *  NEXT_ELEM: To aid removal/insertion, the caller assumes the
 *  responsibility of saving the next element in list
 *  Q_HEAD: Pointer to the head of the queue to iterate through
 *  LINK_NAME: The name of the link used to organize the queue headed
 *  by Q_HEAD.
 **/

#define Q_FOREACH_WITH_MODIFY(CURRENT_ELEM, NEXT_ELEM, Q_HEAD,LINK_NAME) \
    for ((CURRENT_ELEM) = (Q_HEAD)->head;                                \
		((CURRENT_ELEM) != NULL) &&                                      \
		(!!((NEXT_ELEM) = (CURRENT_ELEM)->LINK_NAME.next) | 1);          \
         	(CURRENT_ELEM) = (NEXT_ELEM))

/*
 *  Constructs an iterator block (like a for block) that operates
 *  on each element in Q_HEAD, in backwards order starting from START_ELEM.
 *
 *  Q_FOREACH constructs the head of a block of code that will iterate through
 *  each element in the queue headed by Q_HEAD. Each time through the loop,
 *  the variable named by CURRENT_ELEM will be set to point to a subsequent
 *  element in the queue.
 *
 *  Usage:
 *  Q_FOREACH(CURRENT_ELEM,Q_HEAD,LINK_NAME)
 *  {
 *  ... operate on the variable CURRENT_ELEM ...
 *  }
 *
 *  If LINK_NAME is not used to organize the queue headed by Q_HEAD, then
 *  the behavior of this macro is undefined.
 *
 *  CURRENT_ELEM: name of the variable to use for iteration. On each
 *  loop through the Q_FOREACH block, CURRENT_ELEM will point to the
 *  current element in the queue. CURRENT_ELEM should be an already-
 *  defined variable name, and its type should be a pointer to
 *  the type of data organized by Q_HEAD
 *  START_ELEM: Element to start from
 *  Q_HEAD: Pointer to the head of the queue to iterate through
 *  LINK_NAME: The name of the link used to organize the queue headed
 *  by Q_HEAD.
 */

#define Q_FOREACH_PREV_FROM(CURRENT_ELEM, START_ELEM, Q_HEAD,LINK_NAME)     \
    for ((CURRENT_ELEM) = (START_ELEM)->LINK_NAME.prev;                     \
         (CURRENT_ELEM) != NULL;                                            \
         (CURRENT_ELEM) = (CURRENT_ELEM)->LINK_NAME.prev)



/* Forward declarations */
typedef struct allocator allocator_t;

/* Function declarations */
allocator_t *allocator_init(void *buf, size_t size, size_t alignment);
void *allocator_alloc(allocator_t *ctx, size_t size);
void allocator_free(allocator_t *ctx, void *address);
void allocator_deinit(allocator_t *ctx);

#endif /* __FGPU_INTERNAL_ALLOCATOR_HPP__ */
