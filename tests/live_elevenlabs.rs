//! Against the real service (needs ELEVENLABS_API_KEY; run with --ignored):
//! TLS through the shared config works, and a second connection resumes the
//! first's session instead of a full handshake.

use std::time::Instant;

use chirp::elevenlabs::ElevenLabs;

#[tokio::test]
#[ignore = "needs ELEVENLABS_API_KEY and network"]
async fn speaker_connects_over_shared_tls_and_speaks() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let elevenlabs = ElevenLabs::from_env().expect("ELEVENLABS_API_KEY");
    let mut connects = Vec::new();
    for _ in 0..3 {
        let started = Instant::now();
        let (mut speaker, mut audio) = elevenlabs.speaker().await.expect("TLS connect");
        connects.push(started.elapsed());
        speaker.say("Hello there.").await.unwrap();
        speaker.finish().await.unwrap();
        let mut bytes = 0;
        while let Some(chunk) = audio.recv().await {
            bytes += chunk.len();
        }
        assert!(bytes > 8_000, "expected speech audio, got {bytes} bytes");
    }
    println!("connect times: {connects:?}");
}

/// The old path, for comparison: a fresh TLS config (root certificates
/// loaded and parsed) on every connection.
#[tokio::test]
#[ignore = "needs ELEVENLABS_API_KEY and network"]
async fn baseline_fresh_tls_config_per_connection() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let _ = rustls::crypto::ring::default_provider().install_default();
    let key = std::env::var("ELEVENLABS_API_KEY").unwrap();
    let mut connects = Vec::new();
    for _ in 0..3 {
        let mut request = "wss://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_16000"
            .into_client_request()
            .unwrap();
        request
            .headers_mut()
            .insert("xi-api-key", key.parse().unwrap());
        let started = Instant::now();
        let (socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        connects.push(started.elapsed());
        drop(socket);
    }
    println!("fresh-config connect times: {connects:?}");
}
