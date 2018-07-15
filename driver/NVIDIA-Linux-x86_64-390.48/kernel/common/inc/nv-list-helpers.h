/*******************************************************************************
    Copyright (c) 2013-2016 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/
#ifndef __NV_LIST_HELPERS_H__
#define __NV_LIST_HELPERS_H__

#include <linux/list.h>
#include "conftest.h"

#if !defined (list_for_each)
#define list_for_each(pos, head) \
        for (pos = (head)->next; pos != (head); pos = (pos)->next)
#endif

#if !defined(NV_LIST_CUT_POSITION_PRESENT)
    static inline void __list_cut_position(struct list_head *list,
                                           struct list_head *head,
                                           struct list_head *entry)
    {
         struct list_head *new_first = entry->next;
         list->next = head->next;
         list->next->prev = list;
         list->prev = entry;
         entry->next = list;
         head->next = new_first;
         new_first->prev = head;
    }

    static inline void list_cut_position(struct list_head *list,
                                         struct list_head *head,
                                         struct list_head *entry)
    {
         if (list_empty(head))
             return;
         if ((!list_empty(head) && (head->next == head->prev)) &&
             (head->next != entry && head != entry))
             return;
         if (entry == head)
             INIT_LIST_HEAD(list);
         else
             __list_cut_position(list, head, entry);
    }
#endif

#if !defined(list_first_entry)
    #define list_first_entry(ptr, type, member) \
         list_entry((ptr)->next, type, member)
#endif

#if !defined(list_first_entry_or_null)
    #define list_first_entry_or_null(ptr, type, member) \
        (!list_empty(ptr) ? list_first_entry(ptr, type, member) : NULL)
#endif

#if !defined(list_last_entry)
    #define list_last_entry(ptr, type, member) \
        list_entry((ptr)->prev, type, member)
#endif

#if !defined(list_last_entry_or_null)
    #define list_last_entry_or_null(ptr, type, member) \
        (!list_empty(ptr) ? list_last_entry(ptr, type, member) : NULL)
#endif

#if !defined(list_prev_entry)
    #define list_prev_entry(pos, member) \
        list_entry((pos)->member.prev, typeof(*(pos)), member)
#endif

#if !defined(list_next_entry)
    #define list_next_entry(pos, member) \
        list_entry((pos)->member.next, typeof(*(pos)), member)
#endif

static inline int list_is_first(const struct list_head *list,
                                const struct list_head *head)
{
    return list->prev == head;
}

#endif // __NV_LIST_HELPERS_H__
