//! A phone call over SIP, end to end: a minimal softphone sends an INVITE,
//! streams a recorded question as G.711 μ-law RTP in real time, hears the
//! reply, and hangs up, all over real UDP sockets.

use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration, Instant};

use chirp::call::{Shared, SpeechPolicy};
use chirp::client;
use chirp::elevenlabs::ElevenLabs;
use chirp::endpoint::TurnModel;
use chirp::mock::{self, Script};
use chirp::openai::OpenAiBrain;
use chirp::turn_taking::Policy;
use chirp::vad::VadModel;
use tokio::net::UdpSocket;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The textbook μ-law encoder (G.711, the classic bit-scan form),
/// independent of chirp's.
fn mulaw(sample: i16) -> u8 {
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

/// How the SIP side is set up for a test.
struct Setup {
    rtp_timeout: Duration,
    allow: Option<&'static str>,
    max_calls: usize,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            rtp_timeout: Duration::from_secs(10),
            allow: None,
            max_calls: usize::MAX,
        }
    }
}

async fn chirp_sip(script: Script, setup: Setup) -> SocketAddr {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        webrtc: None,
        admission: chirp::admission::Admission::new(setup.max_calls),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let config = chirp::sip::SipConfig {
        sip_bind: "127.0.0.1:0".parse().unwrap(),
        rtp_bind: "127.0.0.1:0".parse().unwrap(),
        public_ip: None,
        rtp_timeout: setup.rtp_timeout,
        allow: setup
            .allow
            .map(|list| chirp::sip::parse_allow(list).unwrap()),
    };
    let addrs = chirp::sip::start(config, shared, move || {
        OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None)
    })
    .await
    .unwrap();
    addrs.sip
}

/// An INVITE offering PCMU at `media`, in transaction `branch`.
fn invite(
    chirp: SocketAddr,
    from: SocketAddr,
    call_id: &str,
    branch: &str,
    cseq: u32,
    media: u16,
) -> String {
    let sdp = format!(
        "v=0\r\no=phone 1 1 IN IP4 127.0.0.1\r\ns=call\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n\
         m=audio {media} RTP/AVP 0 101\r\na=rtpmap:0 PCMU/8000\r\n"
    );
    format!(
        "INVITE sip:chirp@{chirp} SIP/2.0\r\nVia: SIP/2.0/UDP {from};branch=z9hG4bK{branch}\r\n\
         Max-Forwards: 70\r\nFrom: <sip:phone@127.0.0.1>;tag=caller\r\nTo: <sip:chirp@{chirp}>\r\n\
         Call-ID: {call_id}\r\nCSeq: {cseq} INVITE\r\nContact: <sip:phone@{from}>\r\n\
         Content-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
        sdp.len()
    )
}

/// The final response to a request (skipping 100 Trying), if any comes.
async fn final_response(socket: &UdpSocket, within: Duration) -> Option<String> {
    let mut buffer = vec![0u8; 8192];
    let end = tokio::time::Instant::now() + within;
    loop {
        let (length, _) = tokio::time::timeout_at(end, socket.recv_from(&mut buffer))
            .await
            .ok()?
            .ok()?;
        let text = String::from_utf8_lossy(&buffer[..length]).into_owned();
        if !text.starts_with("SIP/2.0 100") {
            return Some(text);
        }
    }
}

async fn read(socket: &UdpSocket) -> String {
    let mut buffer = vec![0u8; 8192];
    let (length, _) = tokio::time::timeout(Duration::from_secs(5), socket.recv_from(&mut buffer))
        .await
        .expect("a SIP response")
        .unwrap();
    String::from_utf8_lossy(&buffer[..length]).into_owned()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_phone_call_over_sip_is_heard_and_answered() {
    let chirp = chirp_sip(Script::default(), Setup::default()).await;
    let sip = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let rtp = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (sip_local, rtp_local) = (sip.local_addr().unwrap(), rtp.local_addr().unwrap());
    let sdp = format!(
        "v=0\r\no=phone 1 1 IN IP4 127.0.0.1\r\ns=call\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n\
         m=audio {} RTP/AVP 0 101\r\na=rtpmap:0 PCMU/8000\r\n",
        rtp_local.port()
    );
    let headers = |method: &str, cseq: u32, to_tag: &str| {
        format!(
            "Via: SIP/2.0/UDP {sip_local};branch=z9hG4bK{method}{cseq}\r\nMax-Forwards: 70\r\n\
             From: <sip:phone@127.0.0.1>;tag=caller\r\nTo: <sip:chirp@{chirp}>{to_tag}\r\n\
             Call-ID: call-1@127.0.0.1\r\nCSeq: {cseq} {method}\r\nContact: <sip:phone@{sip_local}>\r\n"
        )
    };
    let invite = format!(
        "INVITE sip:chirp@{chirp} SIP/2.0\r\n{}Content-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
        headers("INVITE", 1, ""),
        sdp.len()
    );
    sip.send_to(invite.as_bytes(), chirp).await.unwrap();
    assert!(read(&sip).await.starts_with("SIP/2.0 100"), "no 100 Trying");
    let ok = read(&sip).await;
    assert!(ok.starts_with("SIP/2.0 200"), "INVITE not answered: {ok}");
    let to_tag = ok
        .lines()
        .find(|line| line.starts_with("To:"))
        .and_then(|line| line.split(";tag=").nth(1))
        .expect("a To tag")
        .to_string();
    let media_port: u16 = ok
        .lines()
        .find_map(|line| line.strip_prefix("m=audio "))
        .and_then(|rest| rest.split(' ').next())
        .and_then(|port| port.parse().ok())
        .expect("an answer SDP");
    let media: SocketAddr = format!("127.0.0.1:{media_port}").parse().unwrap();
    let ack = format!(
        "ACK sip:chirp@{chirp} SIP/2.0\r\n{}Content-Length: 0\r\n\r\n",
        headers("ACK", 1, &format!(";tag={to_tag}"))
    );
    sip.send_to(ack.as_bytes(), chirp).await.unwrap();

    // The question at 8 kHz, then silence, in 20 ms μ-law packets.
    let question =
        client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap();
    let narrow: Vec<i16> = question.iter().step_by(2).copied().collect();
    let mut frames = narrow.chunks(160);
    let mut heard = 0usize;
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let mut buffer = vec![0u8; 2048];
    let end = Instant::now() + Duration::from_secs(8);
    let (mut seq, mut timestamp) = (0u16, 0u32);
    while Instant::now() < end {
        tokio::select! {
            _ = tick.tick() => {
                let frame: Vec<u8> = match frames.next() {
                    Some(frame) => frame.iter().map(|s| mulaw(*s)).chain(std::iter::repeat(0xff)).take(160).collect(),
                    None => vec![0xff; 160],
                };
                let mut packet = vec![0x80, 0];
                packet.extend_from_slice(&seq.to_be_bytes());
                packet.extend_from_slice(&timestamp.to_be_bytes());
                packet.extend_from_slice(&0x1234u32.to_be_bytes());
                packet.extend(frame);
                rtp.send_to(&packet, media).await.unwrap();
                seq = seq.wrapping_add(1);
                timestamp = timestamp.wrapping_add(160);
            }
            received = rtp.recv_from(&mut buffer) => {
                let (length, _) = received.unwrap();
                assert_eq!(buffer[1] & 0x7f, 0, "reply not in PCMU");
                heard += length - 12;
            }
        }
    }
    let bye = format!(
        "BYE sip:chirp@{chirp} SIP/2.0\r\n{}Content-Length: 0\r\n\r\n",
        headers("BYE", 2, &format!(";tag={to_tag}"))
    );
    sip.send_to(bye.as_bytes(), chirp).await.unwrap();
    assert!(
        read(&sip).await.starts_with("SIP/2.0 200"),
        "BYE not acknowledged"
    );
    // The mock answers only after a turn ends, which needs the decoded
    // question; then it speaks well over a second.
    assert!(heard > 8_000, "heard only {heard} samples of the reply");
}

/// A caller that vanishes without a BYE is hung up on once its audio stops.
#[tokio::test(flavor = "multi_thread")]
async fn a_caller_who_stops_sending_audio_is_hung_up_on() {
    let chirp = chirp_sip(
        Script::default(),
        Setup {
            rtp_timeout: Duration::from_millis(800),
            ..Setup::default()
        },
    )
    .await;
    let phone = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let from = phone.local_addr().unwrap();
    phone
        .send_to(
            invite(chirp, from, "gone@1", "a", 1, 40_000).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    let ok = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("an answer");
    assert!(ok.starts_with("SIP/2.0 200"), "{ok}");
    // No RTP ever: chirp ends the call and says so.
    let bye = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("a BYE from chirp");
    assert!(bye.starts_with("BYE "), "expected a BYE, got: {bye}");
    assert!(bye.contains("Call-ID: gone@1"), "{bye}");
}

/// Only the trunk's addresses may place calls; others get no answer at all.
#[tokio::test(flavor = "multi_thread")]
async fn a_caller_outside_the_allow_list_gets_no_answer() {
    let chirp = chirp_sip(
        Script::default(),
        Setup {
            allow: Some("10.0.0.0/8"),
            ..Setup::default()
        },
    )
    .await;
    let phone = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let from = phone.local_addr().unwrap();
    phone
        .send_to(
            invite(chirp, from, "outsider@1", "a", 1, 40_002).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    assert_eq!(
        final_response(&phone, Duration::from_millis(800)).await,
        None
    );
}

/// A re-INVITE (a session refresh) is a new transaction: its own 200 OK.
#[tokio::test(flavor = "multi_thread")]
async fn a_re_invite_is_answered_as_its_own_transaction() {
    let chirp = chirp_sip(Script::default(), Setup::default()).await;
    let phone = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let from = phone.local_addr().unwrap();
    phone
        .send_to(
            invite(chirp, from, "refresh@1", "first", 1, 40_004).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    let first = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("an answer");
    // A retransmission of the same transaction gets the same answer.
    phone
        .send_to(
            invite(chirp, from, "refresh@1", "first", 1, 40_004).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    assert_eq!(
        final_response(&phone, Duration::from_secs(5))
            .await
            .as_deref(),
        Some(first.as_str())
    );
    phone
        .send_to(
            invite(chirp, from, "refresh@1", "second", 2, 40_004).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    let refreshed = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("an answer");
    assert!(refreshed.starts_with("SIP/2.0 200"), "{refreshed}");
    assert!(
        refreshed.contains("CSeq: 2 INVITE") && refreshed.contains("branch=z9hG4bKsecond"),
        "{refreshed}"
    );
}

/// A second call claiming the first one's media address can't take it over.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_cannot_take_over_another_calls_media() {
    let chirp = chirp_sip(Script::default(), Setup::default()).await;
    let phone = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let from = phone.local_addr().unwrap();
    phone
        .send_to(
            invite(chirp, from, "first@1", "a", 1, 40_006).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    assert!(final_response(&phone, Duration::from_secs(5))
        .await
        .unwrap()
        .starts_with("SIP/2.0 200"));
    phone
        .send_to(
            invite(chirp, from, "second@1", "b", 1, 40_006).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    let refused = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("an answer");
    assert!(refused.starts_with("SIP/2.0 486"), "{refused}");
}

/// Past the call limit, an INVITE is refused before any session opens.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_over_the_limit_is_refused() {
    let chirp = chirp_sip(
        Script::default(),
        Setup {
            max_calls: 1,
            ..Setup::default()
        },
    )
    .await;
    let phone = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let from = phone.local_addr().unwrap();
    phone
        .send_to(
            invite(chirp, from, "one@1", "a", 1, 40_008).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    assert!(final_response(&phone, Duration::from_secs(5))
        .await
        .unwrap()
        .starts_with("SIP/2.0 200"));
    phone
        .send_to(
            invite(chirp, from, "two@1", "b", 1, 40_010).as_bytes(),
            chirp,
        )
        .await
        .unwrap();
    let refused = final_response(&phone, Duration::from_secs(5))
        .await
        .expect("an answer");
    assert!(refused.starts_with("SIP/2.0 503"), "{refused}");
}
