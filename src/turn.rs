//! Smart Turn v3 end-of-turn predictor, ported from Pipecat 1.13
//! (`pipecat/audio/turn/smart_turn/local_smart_turn_v3.py` + `_whisper_features.py`).
//!
//! `predict` takes the last 8 s of 16 kHz mono audio (left-padded with zeros),
//! computes Whisper-style log-mel features exactly like the vendored numpy
//! code, and runs the ONNX model. The output is a sigmoid probability; > 0.5
//! means the turn is complete.
//!
//! Parity with the Python reference is pinned by `tests/turn_parity.rs`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::TensorRef;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
/// 8 s at 16 kHz.
pub const N_SAMPLES: usize = SAMPLE_RATE * 8;
pub const N_MELS: usize = 80;
/// Frames fed to the model (the 801st STFT frame is dropped, as in the reference).
pub const N_FRAMES: usize = 800;

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = N_FFT / 2 + 1; // 201
const PAD: usize = N_FFT / 2; // centered STFT, reflect padding
const N_STFT_FRAMES: usize = N_FRAMES + 1; // 801
const MEL_FLOOR: f64 = 1e-10;
const NORM_VARIANCE_EPS: f32 = 1e-7;

/// Precomputed Hann window, mel filter bank and FFT plan, plus scratch buffers.
pub struct MelExtractor {
    window: [f64; N_FFT],
    /// Per mel filter: first nonzero FFT bin and the weights from there on.
    filters: Vec<(usize, Vec<f64>)>,
    fft: Arc<dyn Fft<f64>>,
    padded: Vec<f64>,
    buf: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    power: Vec<f64>,
    log_mel: Vec<f64>,
}

fn hertz_to_mel_slaney(f: f64) -> f64 {
    if f >= 1000.0 {
        15.0 + (f / 1000.0).ln() * (27.0 / 6.4f64.ln())
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hertz_slaney(m: f64) -> f64 {
    if m >= 15.0 {
        1000.0 * ((6.4f64.ln() / 27.0) * (m - 15.0)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// `np.linspace(start, stop, num)` (numpy computes `start + i*step`, and pins the last
/// element to `stop`).
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    v[num - 1] = stop;
    v
}

/// Slaney-normalized triangular filter bank, `[mel][bin]` (transpose of the
/// numpy `(num_frequency_bins, num_mel_filters)` matrix).
fn mel_filter_bank() -> Vec<[f64; N_BINS]> {
    let mel_min = hertz_to_mel_slaney(0.0);
    let mel_max = hertz_to_mel_slaney(SAMPLE_RATE as f64 / 2.0);
    let filter_freqs: Vec<f64> = linspace(mel_min, mel_max, N_MELS + 2)
        .into_iter()
        .map(mel_to_hertz_slaney)
        .collect();
    let fft_freqs = linspace(0.0, (SAMPLE_RATE / 2) as f64, N_BINS);
    let diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    (0..N_MELS)
        .map(|m| {
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            let mut row = [0.0; N_BINS];
            for (k, &f) in fft_freqs.iter().enumerate() {
                let down = -(filter_freqs[m] - f) / diff[m];
                let up = (filter_freqs[m + 2] - f) / diff[m + 1];
                row[k] = 0f64.max(down.min(up)) * enorm;
            }
            row
        })
        .collect()
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        // np.hanning(N + 1)[:-1]: periodic Hann.
        let mut window = [0.0; N_FFT];
        for (n, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos();
        }
        let filters = mel_filter_bank()
            .into_iter()
            .map(|row| {
                let lo = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let hi = row.iter().rposition(|&w| w != 0.0).map_or(lo, |i| i + 1);
                (lo, row[lo..hi].to_vec())
            })
            .collect();
        let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        let scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
        Self {
            window,
            filters,
            fft,
            padded: vec![0.0; N_SAMPLES + 2 * PAD],
            buf: vec![Complex::default(); N_FFT],
            scratch,
            power: vec![0.0; N_BINS * N_STFT_FRAMES],
            log_mel: vec![0.0; N_MELS * N_FRAMES],
        }
    }

    /// Writes `N_MELS * N_FRAMES` row-major (`[mel][frame]`) features into `out`.
    /// Mirrors `compute_whisper_log_mel_features(audio, do_normalize=True)`.
    pub fn compute_into(&mut self, audio: &[f32], out: &mut [f32]) {
        assert_eq!(out.len(), N_MELS * N_FRAMES);
        // Reference zero-pads at the end / truncates to the leading 8 s.
        let n = audio.len().min(N_SAMPLES);
        let audio = &audio[..n];

        // Zero-mean unit-variance normalization in f32 (over the padded 8 s buffer;
        // trailing pad zeros count toward mean/var). Stats accumulated in f64 and
        // rounded to f32, which matches numpy's pairwise f32 sums to ~1 ulp.
        let sum: f64 = audio.iter().map(|&x| x as f64).sum();
        let mean = (sum / N_SAMPLES as f64) as f32;
        let sq: f64 = audio
            .iter()
            .map(|&x| (x - mean) as f64)
            .map(|d| d * d)
            .sum::<f64>()
            + (N_SAMPLES - n) as f64 * (mean as f64) * (mean as f64);
        let var = (sq / N_SAMPLES as f64) as f32;
        let std = (var + NORM_VARIANCE_EPS).sqrt();

        // Normalized signal, reflect-padded by PAD on both sides (numpy "reflect"
        // excludes the edge sample), promoted to f64.
        let x = &mut self.padded[PAD..PAD + N_SAMPLES];
        for (i, v) in x.iter_mut().enumerate() {
            let s = if i < n { audio[i] } else { 0.0 };
            *v = ((s - mean) / std) as f64;
        }
        for i in 0..PAD {
            self.padded[PAD - 1 - i] = self.padded[PAD + 1 + i];
            self.padded[PAD + N_SAMPLES + i] = self.padded[PAD + N_SAMPLES - 2 - i];
        }

        // Power spectrogram, two real frames per complex FFT:
        // z = a + i b  =>  A[k] = (Z[k] + conj Z[N-k]) / 2, B[k] = (Z[k] - conj Z[N-k]) / 2i.
        let mut t = 0;
        while t < N_STFT_FRAMES {
            let a = &self.padded[t * HOP..t * HOP + N_FFT];
            let has_b = t + 1 < N_STFT_FRAMES;
            // k indexes four buffers in lockstep; an iterator chain reads worse.
            #[allow(clippy::needless_range_loop)]
            for k in 0..N_FFT {
                let w = self.window[k];
                let im = if has_b {
                    self.padded[(t + 1) * HOP + k] * w
                } else {
                    0.0
                };
                self.buf[k] = Complex::new(a[k] * w, im);
            }
            self.fft
                .process_with_scratch(&mut self.buf, &mut self.scratch);
            for k in 0..N_BINS {
                let z = self.buf[k];
                let zc = self.buf[(N_FFT - k) % N_FFT].conj();
                let fa = (z + zc) * 0.5;
                self.power[t * N_BINS + k] = fa.norm_sqr();
                if has_b {
                    let d = (z - zc) * 0.5;
                    // |d / i|^2 == |d|^2
                    self.power[(t + 1) * N_BINS + k] = d.norm_sqr();
                }
            }
            t += 2;
        }

        // Mel projection + log10 with floor; the last STFT frame is dropped.
        let mut max = f64::NEG_INFINITY;
        for (m, (lo, w)) in self.filters.iter().enumerate() {
            for f in 0..N_FRAMES {
                let p = &self.power[f * N_BINS + lo..f * N_BINS + lo + w.len()];
                let e: f64 = p.iter().zip(w).map(|(p, w)| p * w).sum();
                let l = e.max(MEL_FLOOR).log10();
                max = max.max(l);
                self.log_mel[m * N_FRAMES + f] = l;
            }
        }
        let floor = max - 8.0;
        for (o, &l) in out.iter_mut().zip(&self.log_mel) {
            *o = ((l.max(floor) + 4.0) / 4.0) as f32;
        }
    }

    pub fn compute(&mut self, audio: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; N_MELS * N_FRAMES];
        self.compute_into(audio, &mut out);
        out
    }
}

/// Log-mel features for exactly 8 s of 16 kHz audio (shorter input is zero-padded at
/// the end, longer is truncated, as in the reference). Returns `80 * 800` row-major.
pub fn whisper_log_mel(audio_16k: &[f32]) -> Vec<f32> {
    MelExtractor::new().compute(audio_16k)
}

/// Last 8 s of `audio`, left-padded with zeros (Pipecat's `truncate_audio_to_last_n_seconds`).
pub fn last_8s_left_padded(audio: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), N_SAMPLES);
    let tail = &audio[audio.len().saturating_sub(N_SAMPLES)..];
    let pad = N_SAMPLES - tail.len();
    out[..pad].fill(0.0);
    out[pad..].copy_from_slice(tail);
}

/// Smart Turn v3 predictor. One per worker thread; `predict` is synchronous.
pub struct SmartTurn {
    session: Session,
    mel: MelExtractor,
    window: Vec<f32>,
    features: Vec<f32>,
}

impl SmartTurn {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = crate::onnx::session(model_path)?;
        Ok(Self {
            session,
            mel: MelExtractor::new(),
            window: vec![0.0; N_SAMPLES],
            features: vec![0.0; N_MELS * N_FRAMES],
        })
    }

    /// Features for the model input (last 8 s, left-padded). Exposed for timing/tests.
    pub fn features(&mut self, audio_16k: &[f32]) -> &[f32] {
        last_8s_left_padded(audio_16k, &mut self.window);
        self.mel.compute_into(&self.window, &mut self.features);
        &self.features
    }

    /// Runs the model on precomputed `80 * 800` features. Exposed for timing/tests.
    pub fn infer(&mut self, features: &[f32]) -> Result<f32> {
        let input = TensorRef::from_array_view(([1usize, N_MELS, N_FRAMES], features))
            .map_err(|e| anyhow!("{e}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["input_features" => input])
            .map_err(|e| anyhow!("{e}"))?;
        let (_, probs) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        probs
            .first()
            .copied()
            .ok_or_else(|| anyhow!("empty smart turn output"))
    }

    /// Probability that the turn is complete (> 0.5 means complete).
    pub fn predict(&mut self, audio_16k: &[f32]) -> Result<f32> {
        last_8s_left_padded(audio_16k, &mut self.window);
        self.mel.compute_into(&self.window, &mut self.features);
        let features = std::mem::take(&mut self.features);
        let p = self.infer(&features);
        self.features = features;
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn smart_turn_is_send() {
        assert_send::<SmartTurn>();
    }

    #[test]
    fn silence_is_constant() {
        let f = whisper_log_mel(&vec![0.0; N_SAMPLES]);
        assert!(f.iter().all(|&v| v == -1.5));
    }
}
