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
