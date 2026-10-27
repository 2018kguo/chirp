#![doc = include_str!("../README.md")]

pub mod brain;
pub mod call;
pub mod call_state;
pub mod client;
pub mod elevenlabs;
pub mod endpoint;
#[cfg(feature = "heap-stats")]
pub mod heap;
pub mod mock;
pub mod onnx;
pub mod openai;
pub mod ring;
pub mod server;
pub mod sse;
pub mod turn;
pub mod turn_taking;
pub mod vad;

/// Raise the open-file soft limit to the hard limit, as Go's runtime does.
/// Every call holds several sockets (caller, transcription, speech, brain);
/// at Linux's default soft limit of 1,024, replies past ~300 calls failed to
/// connect and calls silently went unanswered.
pub fn raise_fd_limit() {
    // SAFETY: getrlimit/setrlimit with a valid pointer to a local struct.
    unsafe {
        let mut limit = std::mem::zeroed::<libc::rlimit>();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0 && limit.rlim_cur < limit.rlim_max
        {
            // macOS refuses more than OPEN_MAX for the soft limit.
            #[cfg(target_os = "macos")]
            let max = limit.rlim_max.min(10_240); // OPEN_MAX, sys/syslimits.h
            #[cfg(not(target_os = "macos"))]
            let max = limit.rlim_max;
            limit.rlim_cur = max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
        }
    }
}
