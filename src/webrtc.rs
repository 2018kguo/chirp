//! Calls over WebRTC: Opus audio both ways and a data channel (`events`) for
//! the JSON events. One UDP socket and one task, the hub, serve every call:
//! each packet goes to the peer whose ICE/DTLS session accepts it (str0m is
//! sans-IO, so peers are plain state). The caller's audio is decoded to
//! 16 kHz and handed to `call::run`; replies are paced out in 20 ms Opus
//! frames from a playout queue that an `interrupt` empties.
//!
//! Signaling is one request: `POST /v1/webrtc` with the browser's SDP offer
//! as the body returns the answer (as WHIP does). chirp is ICE-lite with one
//! host candidate, the address in `CHIRP_PUBLIC_ADDR`.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bytes::Bytes;
use opus::{Application, Channels};
use str0m::change::SdpOffer;
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{Frequency, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use crate::brain::Brain;
use crate::call::{self, Shared, ToCaller};

/// 20 ms at 16 kHz: one Opus frame, as calls hear and speak.
const FRAME: usize = 320;
/// The same 20 ms on Opus's RTP clock, which is always 48 kHz.
const FRAME_RTP_TICKS: u64 = 960;

/// The shared socket's hub; cloning shares it.
#[derive(Clone)]
pub struct WebRtc {
    peers: mpsc::Sender<Peer>,
    public: SocketAddr,
}

impl WebRtc {
    /// Binds the call socket on `bind` and starts the hub. `public` is the
    /// address callers send media to (the host candidate); by default the
    /// socket's own.
    pub async fn start(bind: SocketAddr, public: Option<SocketAddr>) -> Result<Self> {
        str0m::crypto::from_feature_flags().install_process_default();
        let socket = UdpSocket::bind(bind)
            .await
            .with_context(|| format!("WebRTC socket on {bind}"))?;
        let public = match public {
            Some(public) => public,
            None => socket.local_addr()?,
        };
        let (peers, incoming) = mpsc::channel(64);
        tokio::spawn(hub(socket, public, incoming));
        Ok(Self { peers, public })
    }

    /// Answers a caller's SDP offer and starts their call.
    pub async fn answer<B: Brain>(&self, offer: &str, shared: Shared, brain: B) -> Result<String> {
        let mut rtc = RtcConfig::new()
            .set_ice_lite(true)
            .clear_codecs()
            .enable_opus(true, false)
            .build(Instant::now());
        rtc.add_local_candidate(Candidate::host(self.public, "udp")?);
        let offer = SdpOffer::from_sdp_string(offer).context("SDP offer")?;
        let answer = rtc.sdp_api().accept_offer(offer)?;
        let mut encoder = opus::Encoder::new(16_000, Channels::Mono, Application::Voip)?;
        // Encoding is most of a WebRTC call's codec cost: per call-second of
        // speech, 5.3 ms at the default complexity (10), 3.5 ms at 5; decoding
        // is 0.3 ms. 5 is the usual server setting for speech.
        encoder.set_complexity(5)?;
        let (audio_in, audio) = mpsc::channel::<Bytes>(64);
        let (to_caller, from_call) = mpsc::channel::<ToCaller>(512);
        let peer = Peer {
            rtc,
            audio_in: Some(audio_in),
            from_call,
            call_ended: false,
            decoder: opus::Decoder::new(16_000, Channels::Mono)?,
            encoder,
            decoded: vec![0; FRAME * 6],
            encoded: vec![0; 1500],
            mid: None,
            pt: None,
            channel: None,
            events: VecDeque::new(),
            playout: VecDeque::new(),
            rtp_time: 0,
        };
        self.peers
            .send(peer)
            .await
            .map_err(|_| anyhow::anyhow!("WebRTC hub stopped"))?;
        let audio = futures::stream::unfold(audio, |mut audio| async move {
            audio.recv().await.map(|pcm| (pcm, audio))
        });
        tokio::spawn(call::run(Box::pin(audio), to_caller, shared, brain));
        Ok(answer.to_sdp_string())
    }
}

struct Peer {
    rtc: Rtc,
    /// The call's audio; dropped (ending the call) when the peer goes away.
    audio_in: Option<mpsc::Sender<Bytes>>,
    from_call: mpsc::Receiver<ToCaller>,
    call_ended: bool,
    decoder: opus::Decoder,
    encoder: opus::Encoder,
    decoded: Vec<i16>,
    encoded: Vec<u8>,
    mid: Option<Mid>,
    pt: Option<Pt>,
    channel: Option<ChannelId>,
    /// Events not yet sent (the data channel may not be open yet).
    events: VecDeque<String>,
    /// Reply audio not yet played, 16 kHz.
    playout: VecDeque<i16>,
    rtp_time: u64,
}

impl Peer {
    /// Sends what the session wants sent; returns when it next needs time.
    fn drive(&mut self, socket: &UdpSocket) -> Instant {
        loop {
            match self.rtc.poll_output() {
                Ok(Output::Transmit(transmit)) => {
                    // A full send buffer drops the packet, as the network would.
                    let _ = socket.try_send_to(&transmit.contents, transmit.destination);
                }
                Ok(Output::Timeout(at)) => return at,
                Ok(Output::Event(event)) => self.event(event),
                Err(error) => {
                    tracing::warn!(%error, "WebRTC session failed");
                    self.rtc.disconnect();
                    return Instant::now() + Duration::from_secs(1);
                }
            }
        }
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
                self.rtc.disconnect();
            }
            Event::MediaAdded(media) => self.mid = Some(media.mid),
            Event::MediaData(data) => {
                let Ok(samples) = self.decoder.decode(&data.data, &mut self.decoded, false) else {
                    return;
                };
                let pcm: Vec<u8> = self.decoded[..samples]
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                if let Some(audio_in) = &self.audio_in {
                    // A call that can't keep up loses audio rather than
                    // stalling every other call on the hub.
                    let _ = audio_in.try_send(Bytes::from(pcm));
                }
            }
            Event::ChannelOpen(id, _) => self.channel = Some(id),
            _ => {}
        }
    }

    /// Every 20 ms: take what the call sent, send events, play one frame.
    fn pace(&mut self, now: Instant) {
        loop {
            match self.from_call.try_recv() {
                Ok(ToCaller::Audio(pcm)) => self.playout.extend(
                    pcm.chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]])),
                ),
                Ok(ToCaller::Event(event)) => {
                    if event["type"] == "interrupt" {
                        self.playout.clear();
                    }
                    self.events.push_back(event.to_string());
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.call_ended = true;
                    break;
                }
            }
        }
        if let Some(id) = self.channel {
            while let Some(event) = self.events.front() {
                match self
                    .rtc
                    .channel(id)
                    .map(|mut channel| channel.write(false, event.as_bytes()))
                {
                    Some(Ok(true)) => {
                        self.events.pop_front();
                    }
                    _ => break,
                }
            }
        }
        self.play_frame(now);
        // The call is over and everything it said has been played.
        if self.call_ended && self.playout.is_empty() && self.events.is_empty() {
            self.rtc.disconnect();
        }
    }

    fn play_frame(&mut self, now: Instant) {
        if self.playout.is_empty() {
            return;
        }
        let Some(mid) = self.mid else { return };
        let mut frame = [0i16; FRAME];
        for (slot, sample) in frame
            .iter_mut()
            .zip(self.playout.drain(..FRAME.min(self.playout.len())))
        {
            *slot = sample;
        }
        let Ok(length) = self.encoder.encode(&frame, &mut self.encoded) else {
            return;
        };
        let Some(writer) = self.rtc.writer(mid) else {
            return;
        };
        let pt = match self.pt {
            Some(pt) => pt,
            None => {
                let Some(pt) = writer
                    .payload_params()
                    .find(|params| params.spec().codec == Codec::Opus)
                    .map(|params| params.pt())
                else {
                    return;
                };
                self.pt = Some(pt);
                pt
            }
        };
        let time = MediaTime::new(self.rtp_time, Frequency::FORTY_EIGHT_KHZ);
        self.rtp_time += FRAME_RTP_TICKS;
        if let Err(error) = writer.write(pt, now, time, self.encoded[..length].to_vec()) {
            tracing::debug!(%error, "dropped an audio frame");
        }
    }
}

/// Serves every WebRTC call on one socket.
async fn hub(socket: UdpSocket, public: SocketAddr, mut incoming: mpsc::Receiver<Peer>) {
    let mut peers: Vec<Peer> = Vec::new();
    let mut buffer = vec![0u8; 2000];
    let mut pace = tokio::time::interval(Duration::from_millis(20));
    pace.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let mut next = Instant::now() + Duration::from_millis(100);
        for peer in &mut peers {
            next = next.min(peer.drive(&socket));
        }
        // A peer that is gone drops its audio sender, which ends its call.
        peers.retain(|peer| peer.rtc.is_alive());
        tokio::select! {
            peer = incoming.recv() => match peer {
                Some(peer) => peers.push(peer),
                None => return,
            },
            received = socket.recv_from(&mut buffer) => {
                let Ok((length, source)) = received else { continue };
                let Ok(contents) = buffer[..length].try_into() else { continue };
                let input = Input::Receive(
                    Instant::now(),
                    Receive { proto: Protocol::Udp, source, destination: public, contents },
                );
                if let Some(peer) = peers.iter_mut().find(|peer| peer.rtc.accepts(&input)) {
                    if let Err(error) = peer.rtc.handle_input(input) {
                        tracing::debug!(%error, "WebRTC input rejected");
                        peer.rtc.disconnect();
                    }
                }
            }
            _ = pace.tick() => {
                let now = Instant::now();
                for peer in &mut peers {
                    peer.pace(now);
                }
            }
            _ = tokio::time::sleep_until(next.into()) => {}
        }
        let now = Instant::now();
        for peer in &mut peers {
            if peer.rtc.is_alive() {
                let _ = peer.rtc.handle_input(Input::Timeout(now));
            }
        }
    }
}
