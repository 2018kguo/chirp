//! A call over WebRTC, end to end: a str0m peer stands in for the browser,
//! offers Opus audio and an `events` data channel, plays a recorded question
//! in real time, and hears the reply, all over a real UDP socket.

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
use chirp::webrtc::WebRtc;
use serde_json::Value;
use str0m::change::SdpAnswer;
use str0m::format::Codec;
use str0m::media::{Direction, Frequency, MediaKind, MediaTime};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, Input, Output, RtcConfig};
use tokio::net::UdpSocket;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// What the caller heard: events from the data channel and reply audio,
/// with their arrival times from the start of the call.
struct Heard {
    events: Vec<(Duration, Value)>,
    audio_at: Vec<Duration>,
    reply_samples: usize,
}

async fn chirp_webrtc(script: Script) -> String {
    let mocks = mock::start(script).await.unwrap();
    let shared = Shared {
        vad: VadModel::spawn(&root().join("models/silero_vad.onnx")).unwrap(),
        turn: TurnModel::spawn_threads(&root().join("models/smart-turn-v3.2-cpu.onnx"), 1).unwrap(),
        elevenlabs: ElevenLabs::new("mock", format!("ws://{mocks}")).unwrap(),
        speech: SpeechPolicy::default(),
        turn_policy: Policy::default(),
        webrtc: Some(
            WebRtc::start("127.0.0.1:0".parse().unwrap(), None)
                .await
                .unwrap(),
        ),
    };
    let http = reqwest::Client::new();
    let brain_url = format!("http://{mocks}/v1");
    let router = chirp::server::router(shared, move || {
        OpenAiBrain::new(http.clone(), &brain_url, "mock", "mock", None)
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/webrtc", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await });
    url
}

/// A browser-like peer: plays `audio` (16 kHz) in real time for `seconds`.
async fn call(url: &str, audio: Vec<i16>, seconds: u64) -> Heard {
    let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let local = socket.local_addr().unwrap();
    let mut rtc = RtcConfig::new()
        .clear_codecs()
        .enable_opus(true, false)
        .build(Instant::now());
    rtc.add_local_candidate(Candidate::host(local, "udp").unwrap());
    let mut change = rtc.sdp_api();
    let mid = change.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None);
    change.add_channel("events".into());
    let (offer, pending) = change.apply().unwrap();
    let response = reqwest::Client::new()
        .post(url)
        .body(offer.to_sdp_string())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 201, "offer refused");
    let answer = SdpAnswer::from_sdp_string(&response.text().await.unwrap()).unwrap();
    rtc.sdp_api().accept_answer(pending, answer).unwrap();

    let mut encoder =
        opus::Encoder::new(16_000, opus::Channels::Mono, opus::Application::Voip).unwrap();
    let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono).unwrap();
    let mut decoded = vec![0i16; 1920];
    let mut packet = vec![0u8; 1500];
    let mut frames = audio.chunks(320);
    let mut rtp_time = 0u64;
    let mut heard = Heard {
        events: Vec::new(),
        audio_at: Vec::new(),
        reply_samples: 0,
    };
    let started = Instant::now();
    let mut buffer = vec![0u8; 2000];
    let mut tick = tokio::time::interval(Duration::from_millis(20));
    let end = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < end {
        let timeout = loop {
            match rtc.poll_output().unwrap() {
                Output::Transmit(t) => {
                    let _ = socket.send_to(&t.contents, t.destination).await;
                }
                Output::Timeout(at) => break at,
                Output::Event(Event::MediaData(data)) => {
                    heard.reply_samples +=
                        decoder.decode(&data.data, &mut decoded, false).unwrap_or(0);
                    heard.audio_at.push(started.elapsed());
                }
                Output::Event(Event::ChannelData(data)) => {
                    if let Ok(event) = serde_json::from_slice(&data.data) {
                        heard.events.push((started.elapsed(), event));
                    }
                }
                Output::Event(_) => {}
            }
        };
        tokio::select! {
            received = socket.recv_from(&mut buffer) => {
                let (length, source) = received.unwrap();
                let input = Input::Receive(Instant::now(), Receive {
                    proto: Protocol::Udp,
                    source,
                    destination: local,
                    contents: buffer[..length].try_into().unwrap(),
                });
                rtc.handle_input(input).unwrap();
            }
            _ = tick.tick() => {
                let frame: Vec<i16> = match frames.next() {
                    Some(frame) => frame.iter().copied().chain(std::iter::repeat(0)).take(320).collect(),
                    None => vec![0; 320],
                };
                // Like a browser's Opus with DTX: silence is (almost) not
                // sent at all, so the server must not wait for it.
                let silent = frame.iter().all(|sample| *sample == 0);
                if let Some(writer) = rtc.writer(mid).filter(|_| !silent) {
                    let pt = writer.payload_params().find(|p| p.spec().codec == Codec::Opus).map(|p| p.pt());
                    if let Some(pt) = pt {
                        let length = encoder.encode(&frame, &mut packet).unwrap();
                        let time = MediaTime::new(rtp_time, Frequency::FORTY_EIGHT_KHZ);
                        let _ = writer.write(pt, Instant::now(), time, packet[..length].to_vec());
                    }
                }
                rtp_time += 960;
            }
            _ = tokio::time::sleep_until(timeout.into()) => {}
        }
        rtc.handle_input(Input::Timeout(Instant::now())).unwrap();
    }
    heard
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_over_webrtc_is_heard_and_answered() {
    let script = Script::default();
    let url = chirp_webrtc(script.clone()).await;
    let question =
        client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap();
    let heard = call(&url, question, 8).await;
    let kinds: Vec<_> = heard
        .events
        .iter()
        .filter_map(|(_, e)| e["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"ready"),
        "no ready event over the data channel: {kinds:?}"
    );
    let transcript = heard
        .events
        .iter()
        .find(|(_, e)| e["type"] == "transcript" && e["final"] == true)
        .unwrap_or_else(|| panic!("no final transcript: {kinds:?}"));
    assert_eq!(transcript.1["text"], script.transcript);
    // The mock speaks 100 ms of audio per text message; well over a second
    // of the reply must have been played back.
    assert!(
        heard.reply_samples > 16_000,
        "heard only {} samples of the reply",
        heard.reply_samples
    );
}

/// Barge-in over WebRTC: chirp paces the reply itself, so once the caller
/// speaks over it, no more of it may be played, including what was queued.
#[tokio::test(flavor = "multi_thread")]
async fn barge_in_over_webrtc_stops_the_reply_at_once() {
    let url = chirp_webrtc(Script {
        reply_words: 300,
        brain_token_gap: Duration::from_millis(2),
        ..Script::default()
    })
    .await;
    let question =
        client::read_wav(&root().join("tests/fixtures/turn/complete_question.wav")).unwrap();
    let mut audio = question.clone();
    audio.extend(std::iter::repeat_n(0i16, 1500 * 16));
    let second_start = audio.len();
    audio.extend(&question);
    let heard = call(&url, audio, 9).await;
    let interrupt = heard
        .events
        .iter()
        .find(|(_, e)| e["type"] == "interrupt")
        .map(|(at, _)| *at)
        .expect("an interrupt");
    let second_turn_end =
        Duration::from_millis(second_start as u64 / 16) + client::speech_end(&question);
    let late: Vec<u128> = heard
        .audio_at
        .iter()
        .filter(|at| **at > interrupt + Duration::from_millis(100) && **at < second_turn_end)
        .map(|at| (*at - interrupt).as_millis())
        .collect();
    assert!(
        heard.audio_at.iter().any(|at| *at < interrupt),
        "the first reply was never heard"
    );
    assert!(
        late.is_empty(),
        "reply audio kept playing after the interrupt: {late:?} ms"
    );
}
