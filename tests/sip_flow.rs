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

async fn chirp_sip(script: Script) -> SocketAddr {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        webrtc: None,
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let addrs = chirp::sip::start(
        "127.0.0.1:0".parse().unwrap(),
        "127.0.0.1:0".parse().unwrap(),
        None,
        shared,
        move || OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None),
    )
    .await
    .unwrap();
    addrs.sip
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
    let chirp = chirp_sip(Script::default()).await;
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
