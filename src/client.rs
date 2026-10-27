//! A scripted caller, for tests and benchmarks: plays 16 kHz mono PCM16 into
//! a call at real-time pace, then silence, and timestamps what comes back.
//! Over a WebSocket (`play`), WebRTC (`play_webrtc`, a browser-like peer) or
//! SIP (`play_sip`, a softphone).

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

pub struct Outcome {
    /// When the scripted speech ended, from the start of the call.
    pub speech_end: Duration,
    /// JSON events with their arrival times.
    pub events: Vec<(Duration, Value)>,
    pub first_audio: Option<Duration>,
    /// Arrival time of every audio message.
    pub audio_at: Vec<Duration>,
    pub audio: Vec<i16>,
    /// When the caller hung up, and when the server closed the connection.
    pub hung_up: Duration,
    pub closed: Duration,
    /// WebRTC only: each reply packet's RTP time (seconds) with its arrival.
    pub rtp: Vec<(Duration, f64)>,
}

impl Outcome {
    /// Arrival time of the first event of `kind` (and matching `filter`).
    pub fn event_at(&self, kind: &str, filter: impl Fn(&Value) -> bool) -> Option<Duration> {
        self.events
            .iter()
            .find(|(_, event)| event["type"] == kind && filter(event))
            .map(|(at, _)| *at)
    }
}

/// Plays `speech`, then `tail` of silence, then hangs up.
pub async fn call(url: &str, speech: &[i16], tail: Duration) -> Result<Outcome> {
    let mut audio = speech.to_vec();
    audio.extend(std::iter::repeat_n(0i16, (tail.as_millis() * 16) as usize));
    let mut outcome = play(url, audio).await?;
    outcome.speech_end = speech_end(speech);
    Ok(outcome)
}

/// Plays `audio` at real-time pace, hangs up, and waits for the server to
/// close the connection.
pub async fn play(url: &str, audio: Vec<i16>) -> Result<Outcome> {
    play_with_backlog(url, audio, Duration::ZERO..Duration::ZERO).await
}

/// As `play`, but the audio in `backlog` (call time) is sent all at once, the
/// way it reaches a server that fell behind and then caught up.
pub async fn play_with_backlog(
    url: &str,
    audio: Vec<i16>,
    backlog: std::ops::Range<Duration>,
) -> Result<Outcome> {
    let (socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .context("connect")?;
    let (mut sink, mut stream) = socket.split();
    let speech_end = speech_end(&audio);
    let frames: Vec<Vec<u8>> = audio
        .chunks(320)
        .map(|frame| frame.iter().flat_map(|s| s.to_le_bytes()).collect())
        .collect();
    let started = Instant::now();
    let sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        for (index, frame) in frames.into_iter().enumerate() {
            if !backlog.contains(&(Duration::from_millis(20) * index as u32)) {
                tick.tick().await;
            }
            if sink.send(Message::Binary(frame.into())).await.is_err() {
                break;
            }
        }
        let hung_up = started.elapsed();
        let _ = sink.close().await;
        hung_up
    });
    let mut outcome = Outcome {
        speech_end,
        events: Vec::new(),
        first_audio: None,
        audio_at: Vec::new(),
        audio: Vec::new(),
        hung_up: Duration::ZERO,
        closed: Duration::ZERO,
        rtp: Vec::new(),
    };
    while let Some(Ok(message)) = stream.next().await {
        let at = started.elapsed();
        match message {
            Message::Binary(pcm) => {
                outcome.first_audio.get_or_insert(at);
                outcome.audio_at.push(at);
                outcome.audio.extend(
                    pcm.chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]])),
                );
            }
            Message::Text(text) => {
                if let Ok(event) = serde_json::from_str(&text) {
                    outcome.events.push((at, event));
                }
            }
            _ => {}
        }
    }
    outcome.closed = started.elapsed();
    outcome.hung_up = sender.await.unwrap_or_default();
    Ok(outcome)
}

pub fn read_wav(path: &std::path::Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path).with_context(|| path.display().to_string())?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "need 16 kHz mono"
    );
    Ok(reader.samples::<i16>().collect::<Result<_, _>>()?)
}

/// Where the speech actually ends: the last sample louder than about -36 dBFS,
/// not the end of the clip (clips carry some trailing quiet).
pub fn speech_end(speech: &[i16]) -> Duration {
    let last = speech
        .iter()
        .rposition(|s| s.unsigned_abs() > 500)
        .unwrap_or(0);
    Duration::from_micros(last as u64 * 1_000_000 / 16_000)
}

/// Plays `audio` over WebRTC to `url` (`POST /v1/webrtc`) like a browser:
/// Opus in 20 ms frames, nothing sent while silent (DTX), events on an
/// `events` data channel. Hangs up when the audio ends.
pub async fn play_webrtc(url: &str, audio: Vec<i16>) -> Result<Outcome> {
    let frames = encode_for_webrtc(&audio)?;
    let count = frames.len();
    play_webrtc_encoded(url, frames, count, speech_end(&audio)).await
}

/// `audio` as the Opus frames a browser would send: `None` for silence,
/// which it doesn't send (DTX). Encode once, play to many callers.
pub fn encode_for_webrtc(audio: &[i16]) -> Result<std::sync::Arc<[Option<bytes::Bytes>]>> {
    let mut encoder = opus::Encoder::new(16_000, opus::Channels::Mono, opus::Application::Voip)?;
    let mut packet = vec![0u8; 1500];
    audio
        .chunks(320)
        .map(|frame| {
            let mut padded = [0i16; 320];
            padded[..frame.len()].copy_from_slice(frame);
            if padded.iter().all(|sample| *sample == 0) {
                return Ok(None);
            }
            let length = encoder.encode(&padded, &mut packet)?;
            Ok(Some(bytes::Bytes::copy_from_slice(&packet[..length])))
        })
        .collect()
}

/// Plays `count` frames of `frames` (cycling) over WebRTC, as `play_webrtc`.
pub async fn play_webrtc_encoded(
    url: &str,
    frames: std::sync::Arc<[Option<bytes::Bytes>]>,
    count: usize,
    speech_end: Duration,
) -> Result<Outcome> {
    use str0m::change::SdpAnswer;
    use str0m::format::Codec;
    use str0m::media::{Direction, Frequency, MediaKind, MediaTime};
    use str0m::net::{Protocol, Receive};
    use str0m::{Candidate, Event, Input, Output, RtcConfig};

    let socket = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
    let port = socket.local_addr()?.port();
    let local = std::net::SocketAddr::new(local_ip(url).await?, port);
    let mut rtc = RtcConfig::new()
        .clear_codecs()
        .enable_opus(true, false)
        .build(Instant::now());
    rtc.add_local_candidate(Candidate::host(local, "udp")?);
    let mut change = rtc.sdp_api();
    let mid = change.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None);
    change.add_channel("events".into());
    let (offer, pending) = change.apply().context("an offer")?;
    let response = reqwest::Client::new()
        .post(url)
        .body(offer.to_sdp_string())
        .send()
        .await?;
    anyhow::ensure!(
        response.status() == 201,
        "offer refused: {}",
        response.status()
    );
    let answer = SdpAnswer::from_sdp_string(&response.text().await?)?;
    rtc.sdp_api().accept_answer(pending, answer)?;

    let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono)?;
    let (mut decoded, mut buffer) = (vec![0i16; 1920], vec![0u8; 2000]);
    let mut outcome = Outcome {
        speech_end,
        events: Vec::new(),
        first_audio: None,
        audio_at: Vec::new(),
        audio: Vec::new(),
        hung_up: Duration::ZERO,
        closed: Duration::ZERO,
        rtp: Vec::new(),
    };
    let mut frames = frames.iter().cycle().take(count);
    let mut rtp_time = 0u64;
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let started = Instant::now();
    loop {
        let timeout = loop {
            match rtc.poll_output()? {
                Output::Transmit(t) => {
                    let _ = socket.send_to(&t.contents, t.destination).await;
                }
                Output::Timeout(at) => break at,
                Output::Event(Event::MediaData(data)) => {
                    let at = started.elapsed();
                    let samples = decoder.decode(&data.data, &mut decoded, false).unwrap_or(0);
                    outcome.first_audio.get_or_insert(at);
                    outcome.audio_at.push(at);
                    outcome.audio.extend_from_slice(&decoded[..samples]);
                    outcome.rtp.push((at, data.time.as_seconds()));
                }
                Output::Event(Event::ChannelData(data)) => {
                    if let Ok(event) = serde_json::from_slice(&data.data) {
                        outcome.events.push((started.elapsed(), event));
                    }
                }
                Output::Event(_) => {}
            }
        };
        tokio::select! {
            received = socket.recv_from(&mut buffer) => {
                let (length, source) = received?;
                let Ok(contents) = buffer[..length].try_into() else { continue };
                rtc.handle_input(Input::Receive(Instant::now(), Receive {
                    proto: Protocol::Udp, source, destination: local, contents,
                }))?;
            }
            _ = tick.tick() => {
                let Some(frame) = frames.next() else { break };
                // Like a browser's Opus with DTX: silence is not sent.
                if let (Some(opus), Some(writer)) = (frame, rtc.writer(mid)) {
                    let pt = writer.payload_params().find(|p| p.spec().codec == Codec::Opus).map(|p| p.pt());
                    if let Some(pt) = pt {
                        let time = MediaTime::new(rtp_time, Frequency::FORTY_EIGHT_KHZ);
                        let _ = writer.write(pt, Instant::now(), time, opus.to_vec());
                    }
                }
                rtp_time += 960;
            }
            _ = tokio::time::sleep_until(timeout.into()) => {}
        }
        rtc.handle_input(Input::Timeout(Instant::now()))?;
    }
    rtc.disconnect();
    outcome.hung_up = started.elapsed();
    outcome.closed = outcome.hung_up;
    Ok(outcome)
}

/// The local address that reaches `url`'s host (WebRTC needs a real one).
async fn local_ip(url: &str) -> Result<std::net::IpAddr> {
    let host = url
        .split("://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .context("a host in the URL")?;
    let target = tokio::net::lookup_host(host)
        .await?
        .next()
        .context("the URL's host")?;
    let probe = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
    probe.connect(target).await?;
    Ok(probe.local_addr()?.ip())
}

/// Plays `audio` as a phone call over SIP to `server` (a SIP UDP address):
/// INVITE offering PCMU, μ-law RTP in 20 ms packets, BYE when the audio ends.
pub async fn play_sip(server: std::net::SocketAddr, audio: Vec<i16>) -> Result<Outcome> {
    let local_ip = {
        let probe = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
        probe.connect(server).await?;
        probe.local_addr()?.ip()
    };
    let sip = tokio::net::UdpSocket::bind((local_ip, 0)).await?;
    let rtp = tokio::net::UdpSocket::bind((local_ip, 0)).await?;
    let (from, media_port) = (sip.local_addr()?, rtp.local_addr()?.port());
    let call_id = format!(
        "{:x}@{local_ip}",
        Instant::now().elapsed().as_nanos() ^ (media_port as u128) << 64
    );
    let request = |method: &str, cseq: u32, to_tag: &str, body: &str| {
        let content = if body.is_empty() {
            String::new()
        } else {
            "Content-Type: application/sdp\r\n".into()
        };
        format!(
            "{method} sip:chirp@{server} SIP/2.0\r\nVia: SIP/2.0/UDP {from};branch=z9hG4bK{method}{cseq}{media_port}\r\n\
             Max-Forwards: 70\r\nFrom: <sip:caller@{local_ip}>;tag=caller{media_port}\r\nTo: <sip:chirp@{server}>{to_tag}\r\n\
             Call-ID: {call_id}\r\nCSeq: {cseq} {method}\r\nContact: <sip:caller@{from}>\r\n{content}\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
    };
    let sdp = format!(
        "v=0\r\no=caller 1 1 IN IP4 {local_ip}\r\ns=call\r\nc=IN IP4 {local_ip}\r\nt=0 0\r\n\
         m=audio {media_port} RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\n"
    );
    sip.send_to(request("INVITE", 1, "", &sdp).as_bytes(), server)
        .await?;
    let mut buffer = vec![0u8; 8192];
    let ok = loop {
        let (length, _) =
            tokio::time::timeout(Duration::from_secs(5), sip.recv_from(&mut buffer)).await??;
        let text = String::from_utf8_lossy(&buffer[..length]).into_owned();
        if !text.starts_with("SIP/2.0 100") {
            break text;
        }
    };
    anyhow::ensure!(
        ok.starts_with("SIP/2.0 200"),
        "INVITE refused: {}",
        ok.lines().next().unwrap_or_default()
    );
    let to_tag = ok
        .lines()
        .find(|line| line.starts_with("To:"))
        .and_then(|line| line.split(";tag=").nth(1))
        .map(|tag| format!(";tag={tag}"))
        .unwrap_or_default();
    let media_port: u16 = ok
        .lines()
        .find_map(|line| line.strip_prefix("m=audio "))
        .and_then(|rest| rest.split(' ').next())
        .and_then(|port| port.parse().ok())
        .context("an answer SDP")?;
    let media = std::net::SocketAddr::new(server.ip(), media_port);
    sip.send_to(request("ACK", 1, &to_tag, "").as_bytes(), server)
        .await?;

    let narrow: Vec<i16> = audio.iter().step_by(2).copied().collect();
    let mut frames = narrow.chunks(160);
    let mut outcome = Outcome {
        speech_end: speech_end(&audio),
        events: Vec::new(),
        first_audio: None,
        audio_at: Vec::new(),
        audio: Vec::new(),
        hung_up: Duration::ZERO,
        closed: Duration::ZERO,
        rtp: Vec::new(),
    };
    let (mut seq, mut timestamp) = (0u16, 0u32);
    let mut packet = Vec::with_capacity(172);
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let started = Instant::now();
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let Some(frame) = frames.next() else { break };
                packet.clear();
                packet.extend_from_slice(&[0x80, 0]);
                packet.extend_from_slice(&seq.to_be_bytes());
                packet.extend_from_slice(&timestamp.to_be_bytes());
                packet.extend_from_slice(&0x1234_5678u32.to_be_bytes());
                packet.extend(frame.iter().map(|sample| mulaw_encode(*sample)));
                packet.resize(12 + 160, 0xff);
                rtp.send_to(&packet, media).await?;
                seq = seq.wrapping_add(1);
                timestamp = timestamp.wrapping_add(160);
            }
            received = rtp.recv_from(&mut buffer) => {
                let (length, _) = received?;
                let at = started.elapsed();
                outcome.first_audio.get_or_insert(at);
                outcome.audio_at.push(at);
                for byte in &buffer[12.min(length)..length] {
                    let sample = mulaw_decode(*byte);
                    outcome.audio.extend([sample, sample]);
                }
            }
        }
    }
    outcome.hung_up = started.elapsed();
    sip.send_to(request("BYE", 2, &to_tag, "").as_bytes(), server)
        .await?;
    let _ = tokio::time::timeout(Duration::from_secs(2), sip.recv_from(&mut buffer)).await;
    outcome.closed = started.elapsed();
    Ok(outcome)
}

/// The textbook μ-law encoder (G.711, the classic bit-scan form), kept
/// independent of the server's.
fn mulaw_encode(sample: i16) -> u8 {
    let mut magnitude = sample as i32;
    let sign = if magnitude < 0 {
        magnitude = -magnitude;
        0x80
    } else {
        0
    };
    magnitude = magnitude.min(32635) + 0x84;
    let (mut exponent, mut mask) = (7, 0x4000);
    while exponent > 0 && magnitude & mask == 0 {
        exponent -= 1;
        mask >>= 1;
    }
    let mantissa = (magnitude >> (exponent + 3)) & 0x0f;
    !((sign | (exponent << 4) | mantissa) as u8)
}

fn mulaw_decode(byte: u8) -> i16 {
    let byte = !byte;
    let magnitude = ((((byte & 0x0f) as i32) << 3) + 0x84) << ((byte >> 4) & 0x07);
    (if byte & 0x80 != 0 {
        0x84 - magnitude
    } else {
        magnitude - 0x84
    }) as i16
}
