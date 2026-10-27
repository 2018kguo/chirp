//! Phone calls over SIP: a minimal user agent server (INVITE, ACK, BYE,
//! OPTIONS) on one UDP socket, and G.711 audio over RTP on another, shared by
//! every call (calls are told apart by the caller's RTP address). The caller's
//! 8 kHz audio is upsampled to 16 kHz for `call::run`; replies are low-passed,
//! downsampled and paced out in 20 ms packets from a playout queue that an
//! `interrupt` empties. Events have nowhere to go on a phone line.
//!
//! A SIP trunk (or a softphone) sends INVITEs to `CHIRP_SIP_PORT`; media
//! flows to `CHIRP_RTP_PORT` at `CHIRP_PUBLIC_ADDR`'s address.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use crate::brain::Brain;
use crate::call::{self, Shared, ToCaller};
use crate::metrics;
use crate::playout::{Encoder, Playout, FRAME};
use crate::request::{CallRequest, Transport};

/// Where the SIP and RTP sockets ended up.
pub struct SipAddrs {
    pub sip: SocketAddr,
    pub rtp: SocketAddr,
}

pub struct SipConfig {
    pub sip_bind: SocketAddr,
    pub rtp_bind: SocketAddr,
    /// The address put in SDP for media; by default the RTP socket's own.
    pub public_ip: Option<IpAddr>,
    /// A caller who sends no audio for this long has gone: chirp hangs up.
    pub rtp_timeout: Duration,
    /// Where requests may come from (addresses or IPv4 CIDR blocks, as
    /// `parse_allow` reads them); others get no answer. `None` accepts any:
    /// every call opens paid speech sessions, so set it to the trunk's.
    pub allow: Option<Vec<(IpAddr, u8)>>,
}

/// `10.0.0.5, 192.0.2.0/24, ::1` as addresses with prefix lengths.
pub fn parse_allow(text: &str) -> Result<Vec<(IpAddr, u8)>> {
    text.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            let (ip, prefix) = item.split_once('/').unwrap_or((item, ""));
            let ip: IpAddr = ip
                .parse()
                .with_context(|| format!("allow-list entry {item}"))?;
            let full = if ip.is_ipv4() { 32 } else { 128 };
            let prefix = if prefix.is_empty() {
                full
            } else {
                prefix.parse()?
            };
            anyhow::ensure!(prefix <= full, "allow-list entry {item}");
            Ok((ip, prefix))
        })
        .collect()
}

fn allowed(ip: IpAddr, list: &[(IpAddr, u8)]) -> bool {
    list.iter().any(|&(net, prefix)| match (ip, net) {
        (IpAddr::V4(ip), IpAddr::V4(net)) => {
            let mask = u32::MAX.checked_shl(32 - prefix as u32).unwrap_or(0);
            u32::from(ip) & mask == u32::from(net) & mask
        }
        (IpAddr::V6(ip), IpAddr::V6(net)) => {
            let mask = u128::MAX.checked_shl(128 - prefix as u32).unwrap_or(0);
            u128::from(ip) & mask == u128::from(net) & mask
        }
        _ => false,
    })
}

/// Starts answering SIP calls, making one brain per call with `make_brain`.
pub async fn start<B, F>(config: SipConfig, shared: Shared, make_brain: F) -> Result<SipAddrs>
where
    B: Brain,
    F: Fn(&CallRequest) -> Option<B> + Send + Sync + 'static,
{
    let sip = UdpSocket::bind(config.sip_bind)
        .await
        .with_context(|| format!("SIP socket on {}", config.sip_bind))?;
    let rtp = UdpSocket::bind(config.rtp_bind)
        .await
        .with_context(|| format!("RTP socket on {}", config.rtp_bind))?;
    let addrs = SipAddrs {
        sip: sip.local_addr()?,
        rtp: rtp.local_addr()?,
    };
    let media = SocketAddr::new(config.public_ip.unwrap_or(addrs.rtp.ip()), addrs.rtp.port());
    let (to_media, from_signaling) = mpsc::channel(64);
    let (ended_tx, ended) = mpsc::unbounded_channel::<String>();
    tokio::spawn(media_hub(rtp, from_signaling, ended_tx, config.rtp_timeout));
    tokio::spawn(signaling(
        sip,
        media,
        config.allow,
        to_media,
        ended,
        shared,
        make_brain,
    ));
    Ok(addrs)
}

// --- signaling ---------------------------------------------------------------

/// A SIP message, as much of it as a user agent server needs.
struct Message {
    start: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Message {
    fn parse(text: &str) -> Option<Self> {
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text, ""));
        let mut lines = head.split("\r\n");
        let start = lines.next()?.to_string();
        let headers = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(name, value)| (expand(name.trim()), value.trim().to_string()))
            .collect();
        Some(Self {
            start,
            headers,
            body: body.to_string(),
        })
    }

    fn method(&self) -> &str {
        self.start.split(' ').next().unwrap_or_default()
    }

    /// The transaction: the top Via's branch.
    fn branch(&self) -> &str {
        self.header("Via")
            .and_then(|via| via.split(";branch=").nth(1))
            .map(|branch| branch.split(';').next().unwrap_or(branch))
            .unwrap_or_default()
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// SIP's compact header names, written out.
fn expand(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "v" => "Via",
        "f" => "From",
        "t" => "To",
        "i" => "Call-ID",
        "m" => "Contact",
        "l" => "Content-Length",
        "c" => "Content-Type",
        _ => name,
    }
    .to_string()
}

/// A response to `request`, echoing what the caller needs to match it.
fn response(
    request: &Message,
    status: &str,
    to_tag: Option<&str>,
    extra: &str,
    body: &str,
) -> String {
    let mut out = format!("SIP/2.0 {status}\r\n");
    for (name, value) in &request.headers {
        if name.eq_ignore_ascii_case("Via") {
            out.push_str(&format!("Via: {value}\r\n"));
        }
    }
    for name in ["From", "Call-ID", "CSeq"] {
        if let Some(value) = request.header(name) {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
    }
    let to = request.header("To").unwrap_or_default();
    match to_tag {
        Some(tag) if !to.contains(";tag=") => out.push_str(&format!("To: {to};tag={tag}\r\n")),
        _ => out.push_str(&format!("To: {to}\r\n")),
    }
    out.push_str(extra);
    out.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    out
}

/// G.711 flavors chirp speaks, by RTP payload type.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Law {
    Mu = 0,
    A = 8,
}

/// The caller's media address and the codec to use, from an SDP offer.
fn read_offer(sdp: &str) -> Option<(SocketAddr, Law)> {
    let mut ip = None;
    let mut audio = None;
    for line in sdp.lines() {
        if let Some(rest) = line.strip_prefix("c=IN IP4 ") {
            ip = rest.trim().parse::<IpAddr>().ok();
        } else if let Some(rest) = line.strip_prefix("m=audio ") {
            let mut fields = rest.split_whitespace();
            let port: u16 = fields.next()?.parse().ok()?;
            let formats: Vec<&str> = fields.skip(1).collect();
            let law = if formats.contains(&"0") {
                Law::Mu
            } else if formats.contains(&"8") {
                Law::A
            } else {
                return None;
            };
            audio = Some((port, law));
        }
    }
    let (port, law) = audio?;
    Some((SocketAddr::new(ip?, port), law))
}

fn answer_sdp(media: SocketAddr, law: Law) -> String {
    let (pt, name) = match law {
        Law::Mu => (0, "PCMU"),
        Law::A => (8, "PCMA"),
    };
    let ip = media.ip();
    format!(
        "v=0\r\no=chirp 1 1 IN IP4 {ip}\r\ns=chirp\r\nc=IN IP4 {ip}\r\nt=0 0\r\n\
         m=audio {} RTP/AVP {pt}\r\na=rtpmap:{pt} {name}/8000\r\na=ptime:20\r\na=sendrecv\r\n",
        media.port()
    )
}

/// An answered call, for retransmitted INVITEs and for hanging up.
struct Dialog {
    /// The last INVITE transaction answered, and its answer: a
    /// retransmission gets the same answer, a re-INVITE a new one.
    branch: String,
    answer: String,
    sdp: String,
    caller: SocketAddr,
    remote_media: SocketAddr,
    /// What a BYE from chirp needs.
    to_tag: String,
    invite: Message,
    cseq: u32,
}

/// RFC 3261's timers: the first retransmission, and the longest gap.
const T1: Duration = Duration::from_millis(500);
const T2: Duration = Duration::from_secs(4);

/// A 200 OK awaiting its ACK.
struct Unacked {
    /// The INVITE's CSeq number: only its ACK (same number) counts.
    cseq: u32,
    next: tokio::time::Instant,
    interval: Duration,
    give_up: tokio::time::Instant,
}

impl Unacked {
    fn new(invite: &Message) -> Self {
        let now = tokio::time::Instant::now();
        Self {
            cseq: cseq_number(invite),
            next: now + T1,
            interval: T1,
            give_up: now + T1 * 64,
        }
    }
}

enum ToMedia {
    Add {
        call_id: String,
        remote: SocketAddr,
        law: Law,
        audio_in: mpsc::Sender<Bytes>,
        from_call: mpsc::Receiver<ToCaller>,
    },
    Remove(SocketAddr),
}

async fn signaling<B, F>(
    socket: UdpSocket,
    media: SocketAddr,
    allow: Option<Vec<(IpAddr, u8)>>,
    to_media: mpsc::Sender<ToMedia>,
    mut ended: mpsc::UnboundedReceiver<String>,
    shared: Shared,
    make_brain: F,
) where
    B: Brain,
    F: Fn(&CallRequest) -> Option<B> + Send + Sync + 'static,
{
    let local = socket.local_addr().ok();
    let contact = format!(
        "Contact: <sip:chirp@{}:{}>\r\n",
        media.ip(),
        local.map_or(5060, |l| l.port())
    );
    let mut dialogs: HashMap<String, Dialog> = HashMap::new();
    // Each call's media address, to its Call-ID.
    let mut by_media: HashMap<SocketAddr, String> = HashMap::new();
    // Answers not yet ACKed (usually none), by Call-ID.
    let mut unacked: HashMap<String, Unacked> = HashMap::new();
    let mut retransmit = tokio::time::interval(T1 / 2);
    // Ticks missed while nothing was pending aren't owed.
    retransmit.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut buffer = vec![0u8; 8192];
    loop {
        tokio::select! {
            received = socket.recv_from(&mut buffer) => {
                let Ok((length, from)) = received else { continue };
                if allow.as_ref().is_some_and(|list| !allowed(from.ip(), list)) {
                    continue; // not the trunk: no answer at all
                }
                let Ok(text) = std::str::from_utf8(&buffer[..length]) else { continue };
                let Some(request) = Message::parse(text) else { continue };
                if request.start.starts_with("SIP/2.0") {
                    continue; // a response, to our BYE
                }
                let call_id = request.header("Call-ID").unwrap_or_default().to_string();
                let reply = match request.method() {
                    "INVITE" => {
                        let offer = read_offer(&request.body);
                        if let Some(dialog) = dialogs.get_mut(&call_id) {
                            if dialog.branch != request.branch() {
                                // A re-INVITE (a session refresh): same call, same media.
                                let extra = format!("{contact}Content-Type: application/sdp\r\n");
                                dialog.answer = response(&request, "200 OK", Some(&dialog.to_tag), &extra, &dialog.sdp);
                                dialog.branch = request.branch().to_string();
                                unacked.insert(call_id.clone(), Unacked::new(&request));
                            }
                            Some(dialog.answer.clone())
                        } else if let Some((remote, law)) = offer {
                            let slot = match by_media.entry(remote) {
                                Entry::Vacant(slot) => slot,
                                // That media address is another call's.
                                Entry::Occupied(_) => {
                                    let busy = response(&request, "486 Busy Here", None, "", "");
                                    let _ = socket.send_to(busy.as_bytes(), from).await;
                                    continue;
                                }
                            };
                            if shared.turn.overloaded() {
                                metrics::REFUSED_OVERLOADED.add();
                                Some(response(&request, "503 Service Unavailable", None, "", ""))
                            } else {
                                let caller = CallRequest {
                                    transport: Transport::Sip,
                                    remote: Some(from),
                                    headers: Default::default(),
                                    query: [("from", "From"), ("to", "To"), ("call_id", "Call-ID")]
                                        .into_iter()
                                        .filter_map(|(key, header)| {
                                            Some((key.to_string(), request.header(header)?.to_string()))
                                        })
                                        .collect(),
                                };
                                match (make_brain(&caller), shared.admission.admit()) {
                                    (None, _) => {
                                        metrics::REFUSED_FORBIDDEN.add();
                                        Some(response(&request, "403 Forbidden", None, "", ""))
                                    }
                                    (Some(_), None) => {
                                        metrics::REFUSED_LIMIT.add();
                                        Some(response(&request, "503 Service Unavailable", None, "", ""))
                                    }
                                    (Some(brain), Some(permit)) => {
                                        let to_tag = format!("{:x}", rand_tag());
                                        let sdp = answer_sdp(media, law);
                                        let extra = format!("{contact}Content-Type: application/sdp\r\n");
                                        let answer = response(&request, "200 OK", Some(&to_tag), &extra, &sdp);
                                        let _ = socket
                                            .send_to(response(&request, "100 Trying", None, "", "").as_bytes(), from)
                                            .await;
                                        let (audio_in, audio) = mpsc::channel::<Bytes>(64);
                                        let (to_caller, from_call) = mpsc::channel::<ToCaller>(512);
                                        let _ = to_media
                                            .send(ToMedia::Add { call_id: call_id.clone(), remote, law, audio_in, from_call })
                                            .await;
                                        let audio = futures::stream::unfold(audio, |mut audio| async move {
                                            audio.recv().await.map(|pcm| (pcm, audio))
                                        });
                                        let shared = shared.clone();
                                        tokio::spawn(async move {
                                            call::run(Box::pin(audio), to_caller, shared, brain).await;
                                            drop(permit);
                                        });
                                        slot.insert(call_id.clone());
                                        unacked.insert(call_id.clone(), Unacked::new(&request));
                                        dialogs.insert(call_id, Dialog {
                                            branch: request.branch().to_string(),
                                            answer: answer.clone(),
                                            sdp,
                                            caller: from,
                                            remote_media: remote,
                                            to_tag,
                                            invite: request,
                                            cseq: 1,
                                        });
                                        Some(answer)
                                    }
                                }
                            }
                        } else {
                            Some(response(&request, "488 Not Acceptable Here", None, "", ""))
                        }
                    }
                    // In-dialog requests must carry chirp's To tag, which only
                    // the caller saw (in the 200 OK): knowing a Call-ID isn't
                    // enough to stop an answer's retransmission or end a call.
                    "ACK" => {
                        let ours = dialogs.get(&call_id).is_some_and(|dialog| has_to_tag(&request, &dialog.to_tag));
                        // A late ACK for an earlier INVITE leaves a
                        // re-INVITE's answer pending.
                        if ours && unacked.get(&call_id).is_some_and(|pending| pending.cseq == cseq_number(&request)) {
                            unacked.remove(&call_id);
                        }
                        None
                    }
                    "BYE" => match dialogs.get(&call_id) {
                        Some(dialog) if has_to_tag(&request, &dialog.to_tag) => {
                            unacked.remove(&call_id);
                            let remote = dialog.remote_media;
                            dialogs.remove(&call_id);
                            by_media.remove(&remote);
                            let _ = to_media.send(ToMedia::Remove(remote)).await;
                            Some(response(&request, "200 OK", None, "", ""))
                        }
                        _ => Some(response(&request, "481 Call/Transaction Does Not Exist", None, "", "")),
                    },
                    // Every INVITE is answered at once, so a CANCEL is always
                    // too late and changes nothing (RFC 3261 9.2); the caller
                    // ends the call with ACK and BYE.
                    "CANCEL" => Some(response(&request, "200 OK", None, "", "")),
                    "OPTIONS" => Some(response(&request, "200 OK", None, &contact, "")),
                    _ => Some(response(&request, "501 Not Implemented", None, "", "")),
                };
                if let Some(reply) = reply {
                    let _ = socket.send_to(reply.as_bytes(), from).await;
                }
            }
            Some(call_id) = ended.recv() => {
                // The call ended on chirp's side: hang up on the caller. By
                // Call-ID, so a late notice can't end a newer call that took
                // over the same media address.
                if let Some(mut dialog) = dialogs.remove(&call_id) {
                    by_media.remove(&dialog.remote_media);
                    unacked.remove(&call_id);
                    dialog.cseq += 1;
                    let bye = bye(&dialog, &call_id, &contact);
                    let _ = socket.send_to(bye.as_bytes(), dialog.caller).await;
                }
            }
            _ = retransmit.tick(), if !unacked.is_empty() => {
                // RFC 3261 13.3.1.4: over UDP a 200 OK is resent, at T1
                // doubling to T2, until its ACK comes; a caller that never
                // ACKs is hung up on after 64 T1.
                let now = tokio::time::Instant::now();
                let mut gave_up = Vec::new();
                for (call_id, pending) in unacked.iter_mut() {
                    let Some(dialog) = dialogs.get(call_id) else {
                        gave_up.push(call_id.clone());
                        continue;
                    };
                    if now >= pending.give_up {
                        gave_up.push(call_id.clone());
                        continue;
                    }
                    if now < pending.next {
                        continue;
                    }
                    let _ = socket.send_to(dialog.answer.as_bytes(), dialog.caller).await;
                    pending.interval = (pending.interval * 2).min(T2);
                    pending.next = now + pending.interval;
                }
                for call_id in gave_up {
                    unacked.remove(&call_id);
                    if let Some(mut dialog) = dialogs.remove(&call_id) {
                        by_media.remove(&dialog.remote_media);
                        let _ = to_media.send(ToMedia::Remove(dialog.remote_media)).await;
                        dialog.cseq += 1;
                        let bye = bye(&dialog, &call_id, &contact);
                        let _ = socket.send_to(bye.as_bytes(), dialog.caller).await;
                    }
                }
            }
        }
    }
}

/// Whether `request`'s To header carries `tag`.
fn has_to_tag(request: &Message, tag: &str) -> bool {
    request
        .header("To")
        .and_then(|to| to.split(";tag=").nth(1))
        .is_some_and(|value| value.split(';').next() == Some(tag))
}

/// A request's CSeq number (`CSeq: 2 INVITE` -> 2).
fn cseq_number(request: &Message) -> u32 {
    request
        .header("CSeq")
        .and_then(|cseq| cseq.split_whitespace().next())
        .and_then(|number| number.parse().ok())
        .unwrap_or_default()
}

/// A BYE in an answered dialog: chirp is the callee, so From and To swap.
fn bye(dialog: &Dialog, call_id: &str, contact: &str) -> String {
    let invite = &dialog.invite;
    let target = invite
        .header("Contact")
        .and_then(|c| c.split('<').nth(1))
        .and_then(|c| c.split('>').next())
        .map(str::to_string)
        .unwrap_or_else(|| format!("sip:{}", dialog.caller));
    let from = invite.header("To").unwrap_or_default();
    let to = invite.header("From").unwrap_or_default();
    format!(
        "BYE {target} SIP/2.0\r\nVia: SIP/2.0/UDP {};branch=z9hG4bK{:x}\r\nMax-Forwards: 70\r\n\
         From: {from};tag={}\r\nTo: {to}\r\nCall-ID: {call_id}\r\nCSeq: {} BYE\r\n{contact}Content-Length: 0\r\n\r\n",
        contact.trim_start_matches("Contact: <sip:chirp@").trim_end_matches(">\r\n"),
        rand_tag(),
        dialog.to_tag,
        dialog.cseq,
    )
}

fn rand_tag() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    hasher.finish()
}

// --- media -------------------------------------------------------------------

struct RtpCall {
    call_id: String,
    law: Law,
    audio_in: mpsc::Sender<Bytes>,
    playout: Playout<Encoder>,
    up: Upsampler,
    seq: u16,
    timestamp: u32,
    ssrc: u32,
    started: bool,
    /// When the caller's audio last arrived (or the call started).
    last_rtp: tokio::time::Instant,
}

/// Every SIP call's audio on one socket.
async fn media_hub(
    socket: UdpSocket,
    mut from_signaling: mpsc::Receiver<ToMedia>,
    ended: mpsc::UnboundedSender<String>,
    rtp_timeout: Duration,
) {
    let mut calls: HashMap<SocketAddr, RtpCall> = HashMap::new();
    let mut buffer = vec![0u8; 2048];
    let mut packet = Vec::with_capacity(12 + FRAME / 2);
    // The caller's audio, written here and split off per packet: no
    // allocation per packet once warm.
    let mut inbound = BytesMut::new();
    let mut pace = tokio::time::interval(Duration::from_millis(20));
    pace.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
    loop {
        tokio::select! {
            command = from_signaling.recv() => match command {
                Some(ToMedia::Add { call_id, remote, law, audio_in, from_call }) => {
                    let mut down = Downsampler::default();
                    let encode: Encoder = Box::new(move |frame: &[i16; FRAME]| {
                        let mut narrow = [0i16; FRAME / 2];
                        down.process(frame, &mut narrow);
                        Some(narrow.iter().map(|sample| encode(law, *sample)).collect::<Vec<u8>>().into())
                    });
                    calls.insert(remote, RtpCall {
                        call_id,
                        law,
                        audio_in,
                        playout: Playout::new(from_call, encode),
                        up: Upsampler::default(),
                        seq: 0,
                        timestamp: 0,
                        ssrc: rand_tag() as u32,
                        started: false,
                        last_rtp: tokio::time::Instant::now(),
                    });
                }
                // Dropping the call's audio sender ends the call.
                Some(ToMedia::Remove(remote)) => { calls.remove(&remote); }
                None => return,
            },
            received = socket.recv_from(&mut buffer) => {
                let Ok((length, source)) = received else { continue };
                let Some(call) = calls.get_mut(&source) else { continue };
                let Some(payload) = rtp_payload(&buffer[..length]) else { continue };
                call.last_rtp = tokio::time::Instant::now();
                inbound.reserve(payload.len() * 4);
                for byte in payload {
                    let [a, b] = call.up.process(decode(call.law, *byte));
                    inbound.extend_from_slice(&a.to_le_bytes());
                    inbound.extend_from_slice(&b.to_le_bytes());
                }
                // A call that can't keep up loses audio rather than stalling the others.
                let _ = call.audio_in.try_send(inbound.split().freeze());
            }
            _ = pace.tick() => {
                let mut finished = Vec::new();
                for (remote, call) in calls.iter_mut() {
                    call.playout.take(|_| {});
                    if let Some(frame) = call.playout.frame() {
                        packet.clear();
                        packet.push(0x80);
                        // Marker bit on the first packet of a talkspurt.
                        packet.push(call.law as u8 | if call.started { 0 } else { 0x80 });
                        packet.extend_from_slice(&call.seq.to_be_bytes());
                        packet.extend_from_slice(&call.timestamp.to_be_bytes());
                        packet.extend_from_slice(&call.ssrc.to_be_bytes());
                        packet.extend_from_slice(&frame);
                        call.started = true;
                        call.seq = call.seq.wrapping_add(1);
                        let _ = socket.try_send_to(&packet, *remote);
                    } else {
                        call.started = false;
                    }
                    // The RTP clock runs whether or not chirp is speaking.
                    call.timestamp = call.timestamp.wrapping_add((FRAME / 2) as u32);
                    // Ended on chirp's side, or the caller went without a BYE.
                    if call.playout.finished() || call.last_rtp.elapsed() > rtp_timeout {
                        finished.push(*remote);
                    }
                }
                for remote in finished {
                    if let Some(call) = calls.remove(&remote) {
                        let _ = ended.send(call.call_id);
                    }
                }
            }
        }
    }
}

/// The payload of an RTP packet (skipping CSRCs, extension and padding).
fn rtp_payload(packet: &[u8]) -> Option<&[u8]> {
    if packet.len() < 12 || packet[0] >> 6 != 2 {
        return None;
    }
    let csrcs = (packet[0] & 0x0f) as usize;
    let mut start = 12 + 4 * csrcs;
    if packet[0] & 0x10 != 0 {
        let words = u16::from_be_bytes([*packet.get(start + 2)?, *packet.get(start + 3)?]) as usize;
        start += 4 + 4 * words;
    }
    let mut end = packet.len();
    if packet[0] & 0x20 != 0 {
        end = end.checked_sub(*packet.last()? as usize)?;
    }
    packet.get(start..end)
}

// --- G.711 and 8 <-> 16 kHz ---------------------------------------------------

fn decode(law: Law, byte: u8) -> i16 {
    match law {
        Law::Mu => {
            let byte = !byte;
            let exponent = (byte >> 4) & 0x07;
            let mantissa = (byte & 0x0f) as i32;
            let magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84;
            if byte & 0x80 != 0 {
                -magnitude as i16
            } else {
                magnitude as i16
            }
        }
        Law::A => {
            let byte = byte ^ 0x55;
            let exponent = (byte >> 4) & 0x07;
            let mantissa = (byte & 0x0f) as i32;
            let magnitude = if exponent == 0 {
                (mantissa << 4) + 8
            } else {
                ((mantissa << 4) + 0x108) << (exponent - 1)
            };
            if byte & 0x80 != 0 {
                magnitude as i16
            } else {
                -magnitude as i16
            }
        }
    }
}

fn encode(law: Law, sample: i16) -> u8 {
    match law {
        Law::Mu => {
            const BIAS: i32 = 0x84;
            let sign = if sample < 0 { 0x80 } else { 0 };
            let magnitude = (sample as i32).abs().min(32635) + BIAS;
            let exponent = (7 - (magnitude.leading_zeros() as i32 - 17)).clamp(0, 7);
            let mantissa = (magnitude >> (exponent + 3)) & 0x0f;
            !(sign | (exponent << 4) as u8 | mantissa as u8)
        }
        Law::A => {
            let sign = if sample >= 0 { 0x80 } else { 0 };
            let magnitude = (sample as i32).abs().min(32767) >> 3;
            let byte = if magnitude < 32 {
                (magnitude >> 1) as u8
            } else {
                let exponent = (31 - magnitude.leading_zeros() as i32 - 4).clamp(1, 7);
                ((exponent as u8) << 4) | ((magnitude >> exponent) & 0x0f) as u8
            };
            (sign | byte) ^ 0x55
        }
    }
}

/// 8 -> 16 kHz by linear interpolation (speech-to-text only needs the band
/// the phone line carried).
#[derive(Default)]
struct Upsampler {
    last: i16,
}

impl Upsampler {
    /// The two 16 kHz samples for one 8 kHz sample.
    fn process(&mut self, sample: i16) -> [i16; 2] {
        let between = ((self.last as i32 + sample as i32) / 2) as i16;
        self.last = sample;
        [between, sample]
    }
}

/// 16 -> 8 kHz: a half-band low-pass (so speech above 4 kHz doesn't fold
/// back as noise), computed only for the samples kept, from the taps that
/// aren't zero.
#[derive(Default)]
struct Downsampler {
    /// The last input samples, for the filter's reach into the past.
    history: [f32; TAPS.len() - 1],
}

/// Half-band low-pass (Hamming-windowed sinc, 11 taps), cutoff 4 kHz at 16 kHz.
const TAPS: [f32; 11] = [
    0.0051, 0.0000, -0.0422, 0.0000, 0.2903, 0.5000, 0.2903, 0.0000, -0.0422, 0.0000, 0.0051,
];

impl Downsampler {
    fn process(&mut self, wide: &[i16; FRAME], narrow: &mut [i16; FRAME / 2]) {
        const REACH: usize = TAPS.len() - 1;
        let gain: f32 = TAPS.iter().sum();
        let mut window = [0f32; REACH + FRAME];
        window[..REACH].copy_from_slice(&self.history);
        for (slot, &sample) in window[REACH..].iter_mut().zip(wide) {
            *slot = sample as f32;
        }
        for (out, odd) in narrow.iter_mut().zip((1..FRAME).step_by(2)) {
            let reach = &window[odd..odd + TAPS.len()];
            let filtered = TAPS
                .iter()
                .zip(reach)
                .filter(|(tap, _)| **tap != 0.0)
                .map(|(tap, x)| tap * x)
                .sum::<f32>()
                / gain;
            *out = filtered.clamp(-32768.0, 32767.0) as i16;
        }
        self.history.copy_from_slice(&window[FRAME..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g711_round_trips_within_its_quantization() {
        for law in [Law::Mu, Law::A] {
            for sample in (-32000i16..32000).step_by(97) {
                let back = decode(law, encode(law, sample));
                // Logarithmic steps: error grows with level, ~1/16 of it at most.
                let allowed = (sample as i32).abs() / 16 + 16;
                assert!(
                    (back as i32 - sample as i32).abs() <= allowed,
                    "{law:?}: {sample} came back as {back}"
                );
            }
        }
    }

    #[test]
    fn an_offer_gives_the_callers_media_address_and_codec() {
        let offer = "v=0\r\no=- 1 1 IN IP4 10.0.0.5\r\ns=-\r\nc=IN IP4 10.0.0.5\r\nt=0 0\r\nm=audio 4000 RTP/AVP 8 0 101\r\n";
        assert_eq!(
            read_offer(offer),
            Some(("10.0.0.5:4000".parse().unwrap(), Law::Mu))
        );
        let a_law = offer.replace("RTP/AVP 8 0 101", "RTP/AVP 8 101");
        assert_eq!(read_offer(&a_law).map(|(_, law)| law), Some(Law::A));
        assert_eq!(
            read_offer(&offer.replace("RTP/AVP 8 0 101", "RTP/AVP 9")),
            None
        );
    }

    #[test]
    fn the_downsampler_passes_speech_and_rejects_what_would_alias() {
        let tone = |hz: f32| -> Vec<i16> {
            (0..1600)
                .map(|i| (8000.0 * (i as f32 * hz * std::f32::consts::TAU / 16000.0).sin()) as i16)
                .collect()
        };
        let rms = |x: &[i16]| {
            (x.iter().skip(20).map(|&s| (s as f32).powi(2)).sum::<f32>() / (x.len() - 20) as f32)
                .sqrt()
        };
        let down = |wide: Vec<i16>| -> Vec<i16> {
            let mut downsampler = Downsampler::default();
            wide.chunks_exact(FRAME)
                .flat_map(|frame| {
                    let mut narrow = [0i16; FRAME / 2];
                    downsampler.process(frame.try_into().unwrap(), &mut narrow);
                    narrow
                })
                .collect()
        };
        let speech = down(tone(1000.0));
        let alias = down(tone(6000.0));
        assert!(rms(&speech) > 5000.0, "1 kHz lost: rms {}", rms(&speech));
        assert!(
            rms(&alias) < 1500.0,
            "6 kHz folded back: rms {}",
            rms(&alias)
        );
    }
}
