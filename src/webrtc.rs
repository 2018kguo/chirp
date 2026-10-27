//! Calls over WebRTC: Opus audio both ways and a data channel (`events`) for
//! the JSON events. Signaling is one request: `POST /v1/webrtc` with the
//! browser's SDP offer as the body returns the answer (as WHIP does). chirp
//! is ICE-lite with one host candidate per hub.
//!
//! Two kinds of task share the work so neither grows with the number of
//! calls:
//! - a **hub** owns one UDP socket and the sessions (str0m, sans-IO) of the
//!   calls assigned to it. A packet goes to the session its source address
//!   belongs to; a new address is admitted only by a STUN check naming one of
//!   the hub's sessions (by ICE username), so stray traffic costs a hash
//!   lookup. Sessions are driven only when a packet, a message or their own
//!   deadline arrives. `CHIRP_RTC_SHARDS` hubs run on consecutive ports.
//! - each call's **media task** decodes the caller's Opus for `call::run`,
//!   fills the gaps browsers leave in silence (DTX), encodes the reply as it
//!   arrives and paces it out every 20 ms; an `interrupt` drops what has not
//!   been played. Codec work runs on every core, not on the hubs.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use opus::{Application, Channels};
use str0m::change::SdpOffer;
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{Frequency, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive};
use str0m::rtp::Extension;
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use crate::admission::CallPermit;
use crate::brain::Brain;
use crate::call::{self, Shared, ToCaller};
use crate::playout::{Encoder, Playout, FRAME};

/// 20 ms on Opus's RTP clock, which is always 48 kHz.
const FRAME_RTP_TICKS: u64 = 960;
/// No caller audio for this long (beyond network jitter) is silence.
const SILENCE_AFTER: Duration = Duration::from_millis(60);
/// Events kept for a caller whose data channel isn't (or never) open.
const MAX_QUEUED_EVENTS: usize = 256;

/// Every WebRTC hub; cloning shares them.
#[derive(Clone)]
pub struct WebRtc {
    hubs: Arc<[HubHandle]>,
    next_hub: Arc<AtomicUsize>,
    next_id: Arc<AtomicU64>,
}

#[derive(Clone)]
struct HubHandle {
    sessions: mpsc::Sender<NewSession>,
    media: mpsc::Sender<(u64, FromMedia)>,
    public: SocketAddr,
}

struct NewSession {
    id: u64,
    /// Boxed (str0m's `Rtc` is ~10 KB): channel slots and map buckets
    /// stay small, and a session moves without copying it.
    rtc: Box<Rtc>,
    ufrag: String,
    to_media: mpsc::Sender<Bytes>,
}

/// From a call's media task to its hub.
enum FromMedia {
    /// An encoded 20 ms frame and its RTP time.
    Frame(Bytes, u64),
    Event(String),
    /// The call is over and everything it said has been sent.
    End,
}

impl WebRtc {
    /// Starts `shards` hubs on consecutive ports from `bind` (any free ports
    /// when its port is 0). `public` is the address callers send media to;
    /// by default each socket's own. Its port is the first hub's; the rest
    /// follow.
    pub async fn start(
        bind: SocketAddr,
        public: Option<SocketAddr>,
        shards: usize,
    ) -> Result<Self> {
        str0m::crypto::from_feature_flags().install_process_default();
        let mut hubs = Vec::new();
        for shard in 0..shards.max(1) {
            let mut address = bind;
            if bind.port() != 0 {
                address.set_port(bind.port() + shard as u16);
            }
            let socket = UdpSocket::bind(address)
                .await
                .with_context(|| format!("WebRTC socket on {address}"))?;
            let local = socket.local_addr()?;
            let public = match public {
                Some(public) => SocketAddr::new(public.ip(), public.port() + shard as u16),
                None => local,
            };
            let (sessions, new_sessions) = mpsc::channel(256);
            let (media, from_media) = mpsc::channel(16_384);
            tokio::spawn(hub(socket, public, new_sessions, from_media));
            hubs.push(HubHandle {
                sessions,
                media,
                public,
            });
        }
        Ok(Self {
            hubs: hubs.into(),
            next_hub: Arc::default(),
            next_id: Arc::default(),
        })
    }

    /// Answers a caller's SDP offer and starts their call, which holds
    /// `permit` (its place under the call limit) until it ends.
    pub async fn answer<B: Brain>(
        &self,
        offer: &str,
        shared: Shared,
        brain: B,
        permit: CallPermit,
    ) -> Result<String> {
        let hub = &self.hubs[self.next_hub.fetch_add(1, Ordering::Relaxed) % self.hubs.len()];
        let mut rtc = Box::new(
            RtcConfig::new()
                .set_ice_lite(true)
                .clear_codecs()
                .enable_opus(true, false)
                // Only the media id: no transport-wide congestion feedback
                // (chirp sends one fixed-rate audio stream), whose reports were
                // most of a session's queued RTCP.
                .clear_extension_map()
                .set_extension(4, Extension::RtpMid)
                .build(Instant::now()),
        );
        rtc.add_local_candidate(Candidate::host(hub.public, "udp")?);
        let offer = SdpOffer::from_sdp_string(offer).context("SDP offer")?;
        let answer = rtc.sdp_api().accept_offer(offer)?.to_sdp_string();
        let ufrag = answer
            .lines()
            .find_map(|line| line.strip_prefix("a=ice-ufrag:"))
            .context("an ICE username in the answer")?
            .trim()
            .to_string();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (to_media, opus_in) = mpsc::channel::<Bytes>(64);
        let (audio_in, audio) = mpsc::channel::<Bytes>(64);
        let (to_caller, from_call) = mpsc::channel::<ToCaller>(512);
        hub.sessions
            .send(NewSession {
                id,
                rtc,
                ufrag,
                to_media,
            })
            .await
            .map_err(|_| anyhow::anyhow!("WebRTC hub stopped"))?;
        let media = media(id, opus_in, audio_in, from_call, hub.media.clone())?;
        let audio = futures::stream::unfold(audio, |mut audio| async move {
            audio.recv().await.map(|pcm| (pcm, audio))
        });
        tokio::spawn(media);
        tokio::spawn(async move {
            call::run(Box::pin(audio), to_caller, shared, brain).await;
            drop(permit);
        });
        Ok(answer)
    }
}

// --- each call's media task ---------------------------------------------------

/// Decodes the caller, encodes and paces the reply. Ends when the session
/// goes (dropping `audio_in` ends the call) or the call ends and its last
/// words have been sent.
fn media(
    id: u64,
    mut opus_in: mpsc::Receiver<Bytes>,
    audio_in: mpsc::Sender<Bytes>,
    from_call: mpsc::Receiver<ToCaller>,
    to_hub: mpsc::Sender<(u64, FromMedia)>,
) -> Result<impl std::future::Future<Output = ()>> {
    let mut decoder = opus::Decoder::new(16_000, Channels::Mono)?;
    let mut encoder = opus::Encoder::new(16_000, Channels::Mono, Application::Voip)?;
    // Encoding is most of a WebRTC call's codec cost: per call-second of
    // speech, 5.3 ms at the default complexity (10), 3.5 ms at 5; decoding
    // is 0.3 ms. 5 is the usual server setting for speech.
    encoder.set_complexity(5)?;
    let mut packet = [0u8; 1500];
    let encode: Encoder = Box::new(move |frame: &[i16; FRAME]| {
        let length = encoder.encode(frame, &mut packet).ok()?;
        Some(Bytes::copy_from_slice(&packet[..length]))
    });
    let mut playout = Playout::new(from_call, encode);
    Ok(async move {
        let mut decoded = vec![0i16; FRAME * 6];
        // The caller's audio, written here and split off per packet.
        let mut pcm = BytesMut::new();
        let mut last_media: Option<Instant> = None;
        // The caller's encoder said it went quiet: a header-only Opus packet
        // starts DTX (browsers send the first, then nothing).
        let mut in_dtx = false;
        let mut filled_until = Instant::now();
        let mut rtp_time = 0u64;
        let mut events = Vec::new();
        let mut pace = tokio::time::interval(Duration::from_millis(20));
        pace.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
        loop {
            tokio::select! {
                opus = opus_in.recv() => {
                    let Some(opus) = opus else { return }; // the session is gone
                    let Ok(samples) = decoder.decode(&opus, &mut decoded, false) else { continue };
                    pcm.reserve(samples * 2);
                    for sample in &decoded[..samples] {
                        pcm.extend_from_slice(&sample.to_le_bytes());
                    }
                    // A call that can't keep up loses audio rather than
                    // queueing it without bound.
                    let _ = audio_in.try_send(pcm.split().freeze());
                    last_media = Some(Instant::now());
                    in_dtx = opus.len() <= 2;
                }
                _ = pace.tick() => {
                    let now = Instant::now();
                    // Browsers send (almost) nothing while the caller is
                    // silent (Opus DTX), but a turn ends only after enough
                    // quiet audio: fill the gap, in real time.
                    if let Some(last) = last_media {
                        let from = last.max(filled_until);
                        let gap = now.saturating_duration_since(from);
                        // In DTX the silence is known: fill it at once,
                        // without waiting out network jitter.
                        let wait = if in_dtx { Duration::ZERO } else { SILENCE_AFTER };
                        if gap >= wait {
                            let frames = (gap.as_millis() / 20) as usize;
                            pcm.resize(FRAME * 2 * frames, 0);
                            let _ = audio_in.try_send(pcm.split().freeze());
                            filled_until = from + Duration::from_millis(20) * frames as u32;
                        }
                    }
                    playout.take(|event| events.push(event.to_string()));
                    for event in events.drain(..) {
                        let _ = to_hub.try_send((id, FromMedia::Event(event)));
                    }
                    if let Some(frame) = playout.frame() {
                        let _ = to_hub.try_send((id, FromMedia::Frame(frame, rtp_time)));
                    }
                    // The RTP clock is real time, speaking or not, so the
                    // browser's jitter buffer sees a pause as a pause.
                    rtp_time += FRAME_RTP_TICKS;
                    if playout.finished() {
                        let _ = to_hub.send((id, FromMedia::End)).await;
                        return;
                    }
                }
            }
        }
    })
}

// --- hubs ---------------------------------------------------------------------

struct Session {
    rtc: Box<Rtc>,
    ufrag: String,
    to_media: mpsc::Sender<Bytes>,
    mid: Option<Mid>,
    pt: Option<Pt>,
    channel: Option<ChannelId>,
    /// Events not yet sent (the data channel may not be open yet).
    events: VecDeque<String>,
    /// Source addresses routed to this session, removed with it.
    sources: Vec<SocketAddr>,
    /// This session's entry in the deadline set, if it has one.
    scheduled: Option<Instant>,
}

impl Session {
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
                    return Instant::now();
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
                // Decoding is the media task's; a full queue drops the packet.
                let _ = self.to_media.try_send(Bytes::copy_from_slice(&data.data));
            }
            Event::ChannelOpen(id, _) => {
                self.channel = Some(id);
                self.flush_events();
            }
            _ => {}
        }
    }

    fn flush_events(&mut self) {
        let Some(id) = self.channel else {
            // Without a data channel, events have nowhere to go: keep the latest.
            while self.events.len() > MAX_QUEUED_EVENTS {
                self.events.pop_front();
            }
            return;
        };
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

    fn send(&mut self, message: FromMedia) {
        match message {
            FromMedia::Frame(frame, rtp_time) => {
                let Some(mid) = self.mid else { return };
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
                let time = MediaTime::new(rtp_time, Frequency::FORTY_EIGHT_KHZ);
                if let Err(error) = writer.write(pt, Instant::now(), time, frame.to_vec()) {
                    tracing::debug!(%error, "dropped an audio frame");
                }
            }
            FromMedia::Event(event) => {
                self.events.push_back(event);
                self.flush_events();
            }
            FromMedia::End => {
                self.flush_events();
                self.rtc.disconnect();
            }
        }
    }
}

/// The local ICE username a STUN binding request is addressed to.
fn stun_target_ufrag(packet: &[u8]) -> Option<&str> {
    // A STUN message: type, length, the magic cookie, a transaction id,
    // then attributes (type, length, value padded to 4 bytes).
    if packet.len() < 20 || packet[0] & 0xc0 != 0 || packet[4..8] != [0x21, 0x12, 0xa4, 0x42] {
        return None;
    }
    let mut at = 20;
    while at + 4 <= packet.len() {
        let kind = u16::from_be_bytes([packet[at], packet[at + 1]]);
        let length = u16::from_be_bytes([packet[at + 2], packet[at + 3]]) as usize;
        let value = packet.get(at + 4..at + 4 + length)?;
        if kind == 0x0006 {
            // USERNAME is "<the receiver's ufrag>:<the sender's ufrag>".
            return std::str::from_utf8(value).ok()?.split(':').next();
        }
        at += 4 + length.div_ceil(4) * 4;
    }
    None
}

async fn hub(
    socket: UdpSocket,
    public: SocketAddr,
    mut new_sessions: mpsc::Receiver<NewSession>,
    mut from_media: mpsc::Receiver<(u64, FromMedia)>,
) {
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    let mut routes: HashMap<SocketAddr, u64> = HashMap::new();
    let mut by_ufrag: HashMap<String, u64> = HashMap::new();
    // Each session's next deadline, exactly one entry per session: moving a
    // deadline replaces its entry, so this stays the size of the session set.
    let mut deadlines: BTreeSet<(Instant, u64)> = BTreeSet::new();
    let mut buffer = vec![0u8; 2000];
    loop {
        let next = deadlines
            .first()
            .map_or_else(|| Instant::now() + Duration::from_secs(1), |(at, _)| *at);
        let touched = tokio::select! {
            session = new_sessions.recv() => {
                let Some(new) = session else { return };
                by_ufrag.insert(new.ufrag.clone(), new.id);
                sessions.insert(new.id, Session {
                    rtc: new.rtc,
                    ufrag: new.ufrag,
                    to_media: new.to_media,
                    mid: None,
                    pt: None,
                    channel: None,
                    events: VecDeque::new(),
                    sources: Vec::new(),
                    scheduled: None,
                });
                Some(new.id)
            }
            received = socket.recv_from(&mut buffer) => {
                let Ok((length, source)) = received else { continue };
                let packet = &buffer[..length];
                let Ok(contents) = packet.try_into() else { continue };
                let input = Input::Receive(
                    Instant::now(),
                    Receive { proto: Protocol::Udp, source, destination: public, contents },
                );
                let id = match routes.get(&source) {
                    Some(id) => Some(*id),
                    None => stun_target_ufrag(packet).and_then(|ufrag| by_ufrag.get(ufrag).copied()),
                };
                let Some(session) = id.and_then(|id| sessions.get_mut(&id)) else { continue };
                if !session.rtc.accepts(&input) {
                    continue;
                }
                if routes.insert(source, id.unwrap_or_default()).is_none() {
                    session.sources.push(source);
                }
                if let Err(error) = session.rtc.handle_input(input) {
                    tracing::debug!(%error, "WebRTC input rejected");
                    session.rtc.disconnect();
                }
                id
            }
            message = from_media.recv() => {
                let Some((id, message)) = message else { return };
                if let Some(session) = sessions.get_mut(&id) {
                    session.send(message);
                }
                Some(id)
            }
            _ = tokio::time::sleep_until(next.into()) => {
                let now = Instant::now();
                while let Some(&(at, id)) = deadlines.first() {
                    if at > now {
                        break;
                    }
                    deadlines.pop_first();
                    let Some(session) = sessions.get_mut(&id) else { continue };
                    session.scheduled = None;
                    if session.rtc.is_alive() {
                        let _ = session.rtc.handle_input(Input::Timeout(now));
                    }
                    reschedule(id, session, &socket, &mut deadlines);
                    if !session.rtc.is_alive() {
                        remove(id, &mut sessions, &mut routes, &mut by_ufrag, &mut deadlines);
                    }
                }
                None
            }
        };
        if let Some(id) = touched {
            if let Some(session) = sessions.get_mut(&id) {
                reschedule(id, session, &socket, &mut deadlines);
                if !session.rtc.is_alive() {
                    remove(
                        id,
                        &mut sessions,
                        &mut routes,
                        &mut by_ufrag,
                        &mut deadlines,
                    );
                }
            }
        }
    }
}

fn reschedule(
    id: u64,
    session: &mut Session,
    socket: &UdpSocket,
    deadlines: &mut BTreeSet<(Instant, u64)>,
) {
    let at = session.drive(socket);
    if session.scheduled != Some(at) {
        if let Some(previous) = session.scheduled.replace(at) {
            deadlines.remove(&(previous, id));
        }
        deadlines.insert((at, id));
    }
}

/// Drops a finished session: its media task sees the queue close and ends
/// the call.
fn remove(
    id: u64,
    sessions: &mut HashMap<u64, Session>,
    routes: &mut HashMap<SocketAddr, u64>,
    by_ufrag: &mut HashMap<String, u64>,
    deadlines: &mut BTreeSet<(Instant, u64)>,
) {
    if let Some(session) = sessions.remove(&id) {
        if let Some(at) = session.scheduled {
            deadlines.remove(&(at, id));
        }
        for source in &session.sources {
            routes.remove(source);
        }
        by_ufrag.remove(&session.ufrag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stun_check_names_the_session_it_is_for() {
        let mut packet = vec![0x00, 0x01, 0x00, 0x00, 0x21, 0x12, 0xa4, 0x42];
        packet.extend_from_slice(&[7; 12]); // transaction id
        let username = b"srvr:clnt";
        packet.extend_from_slice(&0x0006u16.to_be_bytes());
        packet.extend_from_slice(&(username.len() as u16).to_be_bytes());
        packet.extend_from_slice(username);
        packet.extend_from_slice(&[0; 3]); // padding to 4 bytes
        assert_eq!(stun_target_ufrag(&packet), Some("srvr"));
        // DTLS, RTP and garbage are not STUN.
        assert_eq!(stun_target_ufrag(&[0x16; 40]), None);
        assert_eq!(stun_target_ufrag(&[0x80; 40]), None);
        assert_eq!(stun_target_ufrag(&packet[..10]), None);
    }
}
