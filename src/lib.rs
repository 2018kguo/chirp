//! chirp: a voice engine that serves many calls per process.
//!
//! Each call is a WebSocket carrying 16 kHz mono PCM16 in and out. Tokio does
//! only I/O; model inference runs on dedicated threads, where one Silero VAD
//! session per thread scores every call's audio in batches (`vad`) and Smart
//! Turn decides end of turn when the caller pauses (`endpoint`, `turn`).
//! Speech-to-text and text-to-speech are streaming services (`elevenlabs`);
//! the answer comes from a [`brain::Brain`].

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
