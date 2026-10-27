//! How many calls run at once, across every transport. Each call opens paid
//! speech sessions the moment it starts, so a flood of offers or INVITEs is
//! refused here, before anything is opened.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct Admission {
    active: Arc<AtomicUsize>,
    max: usize,
    draining: Arc<AtomicBool>,
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
            draining: Arc::default(),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    /// A place for one more call, if there is one.
    pub fn admit(&self) -> Option<CallPermit> {
        if self.draining.load(Ordering::Acquire) {
            return None;
        }
        if self.active.fetch_add(1, Ordering::AcqRel) >= self.max {
            self.active.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(CallPermit(self.active.clone()))
    }

    pub fn active(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    /// Refuses new calls from now on, then waits up to `timeout` for the
    /// calls in progress to end. Returns how many are still running.
    pub async fn drain(&self, timeout: std::time::Duration) -> usize {
        self.draining.store(true, Ordering::Release);
        let deadline = tokio::time::Instant::now() + timeout;
        // ponytail: polls; a Notify on permit drop if drains ever need to be exact.
        while self.active() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        self.active()
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

    #[tokio::test(start_paused = true)]
    async fn a_drain_refuses_new_calls_and_waits_for_the_rest() {
        let admission = Admission::new(4);
        let call = admission.admit().expect("a call");
        let draining = admission.clone();
        let drain =
            tokio::spawn(async move { draining.drain(std::time::Duration::from_secs(30)).await });
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert!(admission.admit().is_none(), "a new call during a drain");
        assert!(!drain.is_finished(), "the drain left a call running");
        drop(call);
        assert_eq!(drain.await.unwrap(), 0);
        // A call that never ends is left after the timeout.
        let stuck = Admission::new(1);
        let _call = stuck.admit();
        assert_eq!(stuck.drain(std::time::Duration::from_secs(5)).await, 1);
    }
}
