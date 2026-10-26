//! A counting allocator (feature `heap-stats`): live heap bytes, so memory
//! per call can be measured exactly rather than inferred from RSS.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Allocation calls ever made (allocs and reallocs): the churn that turns
/// into fragmentation and resident memory the live count doesn't show.
static CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(new_size, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        moved
    }
}

/// Bytes currently allocated through the global allocator.
pub fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Allocation calls so far.
pub fn allocation_calls() -> usize {
    CALLS.load(Ordering::Relaxed)
}
