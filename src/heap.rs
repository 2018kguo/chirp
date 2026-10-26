//! A counting allocator (feature `heap-stats`): live heap bytes, so memory
//! per call can be measured exactly rather than inferred from RSS.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts through to the allocator it wraps (System, mimalloc, jemalloc).
pub struct Counting<A>(pub A);

static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Allocation calls ever made (allocs and reallocs): the churn that turns
/// into fragmentation and resident memory the live count doesn't show.
static CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { self.0.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { self.0.realloc(pointer, layout, new_size) };
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
