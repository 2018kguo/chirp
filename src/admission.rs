//! How many calls run at once, across every transport. Each call opens paid
//! speech sessions the moment it starts, so a flood of offers or INVITEs is
//! refused here, before anything is opened.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct Admission {
    active: Arc<AtomicUsize>,
    max: usize,
}

/// Held for a call's lifetime; dropping it frees the place.
pub struct CallPermit(Arc<AtomicUsize>);

impl Drop for CallPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Admission {
    pub fn new(max: usize) -> Self {
        Self {
            active: Arc::default(),
            max,
        }
    }

    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    /// A place for one more call, if there is one.
    pub fn admit(&self) -> Option<CallPermit> {
        if self.active.fetch_add(1, Ordering::AcqRel) >= self.max {
            self.active.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(CallPermit(self.active.clone()))
    }

    pub fn active(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_up_to_the_limit_and_frees_places_when_calls_end() {
        let admission = Admission::new(2);
        let first = admission.admit().expect("first");
        let _second = admission.admit().expect("second");
        assert!(admission.admit().is_none(), "a third call over the limit");
        assert_eq!(admission.active(), 2);
        drop(first);
        assert!(
            admission.admit().is_some(),
            "a place freed when a call ended"
        );
    }
}
