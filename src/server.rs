//! The HTTP surface: `GET /health` and `GET /v1/calls` (WebSocket upgrade).

use axum::extract::{State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::brain::Brain;
use crate::call::{self, Shared, SpeechPolicy};
use crate::elevenlabs::ElevenLabs;
use crate::endpoint::TurnModel;
use crate::vad::VadModel;

/// Serves calls, making one brain per call with `make_brain`.
pub fn router<B, F>(shared: Shared, make_brain: F) -> Router
where
    B: Brain,
    F: Fn() -> B + Clone + Send + Sync + 'static,
{
    let webrtc_brain = make_brain.clone();
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/v1/calls",
            get(
                move |State(shared): State<Shared>, upgrade: WebSocketUpgrade| {
                    let brain = make_brain();
                    async move {
                        // Behind already: a new call would make every call worse.
                        if shared.turn.overloaded() {
                            return (StatusCode::SERVICE_UNAVAILABLE, "at capacity")
                                .into_response();
                        }
                        let Some(permit) = shared.admission.admit() else {
                            return (StatusCode::SERVICE_UNAVAILABLE, "at the call limit")
                                .into_response();
                        };
                        // Callers send 640-byte frames: small buffers, and a cap
                        // so no one can make the server hold a huge message.
                        upgrade
                            .read_buffer_size(4 * 1024)
                            .write_buffer_size(0)
                            .max_message_size(64 * 1024)
                            .on_upgrade(move |socket| async move {
                                call::run_websocket(socket, shared, brain).await;
                                drop(permit);
                            })
                            .into_response()
                    }
                },
            ),
        )
        .route(
            "/v1/webrtc",
            post(move |State(shared): State<Shared>, offer: String| {
                let brain = webrtc_brain();
                async move {
                    let Some(webrtc) = shared.webrtc.clone() else {
                        return (StatusCode::NOT_FOUND, "WebRTC is off").into_response();
                    };
                    if shared.turn.overloaded() {
                        return (StatusCode::SERVICE_UNAVAILABLE, "at capacity").into_response();
                    }
                    let Some(permit) = shared.admission.admit() else {
                        return (StatusCode::SERVICE_UNAVAILABLE, "at the call limit")
                            .into_response();
                    };
                    match webrtc.answer(&offer, shared, brain, permit).await {
                        Ok(answer) => (
                            StatusCode::CREATED,
                            [(axum::http::header::CONTENT_TYPE, "application/sdp")],
                            answer,
                        )
                            .into_response(),
                        Err(error) => (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
                    }
                }
            }),
        )
        .with_state(shared)
}

/// A listener whose connections send small writes at once (no Nagle): a
/// call's audio leaves in 640-byte frames every 20 ms.
pub fn nodelay(
    listener: tokio::net::TcpListener,
) -> impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = std::net::SocketAddr> {
    use axum::serve::ListenerExt;
    listener.tap_io(|connection| {
        let _ = connection.set_nodelay(true);
    })
}

/// Runs a chirp server configured from the environment (variables in the
/// README), making one brain per call with `make_brain`. Downloads missing
/// models first; returns after SIGTERM, once accepting stops.
pub async fn serve<B, F>(make_brain: F) -> anyhow::Result<()>
where
    B: Brain,
    F: Fn() -> B + Clone + Send + Sync + 'static,
{
    let shared = shared_from_env().await?;
    sip_from_env(&shared, make_brain.clone()).await?;
    listen(router(shared, make_brain)).await
}

/// Answers phone calls when `CHIRP_SIP_PORT` is set (media on
/// `CHIRP_SIP_RTP_PORT`, default 7864); a no-op otherwise.
pub async fn sip_from_env<B, F>(shared: &Shared, make_brain: F) -> anyhow::Result<()>
where
    B: Brain,
    F: Fn() -> B + Send + Sync + 'static,
{
    let Ok(port) = std::env::var("CHIRP_SIP_PORT") else {
        return Ok(());
    };
    let allow = match std::env::var("CHIRP_SIP_ALLOW") {
        Ok(list) => Some(crate::sip::parse_allow(&list)?),
        Err(_) => {
            tracing::warn!("CHIRP_SIP_ALLOW is unset: any address can place calls (each opens paid speech sessions)");
            None
        }
    };
    let config = crate::sip::SipConfig {
        sip_bind: ([0, 0, 0, 0], port.parse()?).into(),
        rtp_bind: ([0, 0, 0, 0], env_or("CHIRP_SIP_RTP_PORT", 7864u16)).into(),
        public_ip: Some(public_ip().await?),
        rtp_timeout: std::time::Duration::from_secs(env_or("CHIRP_SIP_RTP_TIMEOUT_SECS", 10)),
        allow,
    };
    let sip = crate::sip::start(config, shared.clone(), make_brain).await?;
    tracing::info!(sip = %sip.sip, rtp = %sip.rtp, "SIP listening");
    Ok(())
}

/// The address callers reach this host at: `CHIRP_PUBLIC_ADDR`'s, else the
/// outbound one.
async fn public_ip() -> anyhow::Result<std::net::IpAddr> {
    Ok(match std::env::var("CHIRP_PUBLIC_ADDR") {
        Ok(address) => address.parse::<SocketAddr>()?.ip(),
        Err(_) => outbound_ip().await,
    })
}

/// The models, speech services and policies `serve` uses, from the
/// environment; for mounting [`router`] beside your own routes.
pub async fn shared_from_env() -> anyhow::Result<Shared> {
    crate::raise_fd_limit();
    let _ = rustls::crypto::ring::default_provider().install_default();
    let models =
        PathBuf::from(std::env::var("CHIRP_MODELS_DIR").unwrap_or_else(|_| "models".into()));
    crate::models::ensure(&models).await?;
    // Threads that have nothing to do cost nothing; too few cap the box.
    // On a 16-vCPU c7a, half the cores for Smart Turn refused 388 of 1,000
    // calls; all but two served all 1,000 (PLAN.md).
    let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
    Ok(Shared {
        vad: VadModel::spawn_threads(
            &models.join(crate::models::VAD),
            // ~1 core per 1,000 calls, but 2 threads ran at their limit
            // there (p99 frame wait 23 ms vs 5.5 ms with 4).
            env_or("CHIRP_VAD_THREADS", (cores / 4).max(2)),
        )?,
        turn: TurnModel::spawn_threads(
            &models.join(crate::models::TURN),
            // Smart Turn is the throughput limit (~38 ms per decision on a c7a).
            env_or("CHIRP_TURN_THREADS", cores.saturating_sub(2).max(1)),
        )?,
        elevenlabs: ElevenLabs::from_env()?,
        speech: SpeechPolicy::default(),
        turn_policy: crate::turn_taking::Policy::default(),
        webrtc: webrtc_from_env().await?,
        // ~62 calls per vCPU served cleanly on a c7a (PLAN.md); past the
        // limit new calls are refused before they open any paid session.
        admission: crate::admission::Admission::new(env_or("CHIRP_MAX_CALLS", cores * 60)),
    })
}

/// WebRTC media on UDP `CHIRP_RTC_PORT` (default 7862), reached at
/// `CHIRP_PUBLIC_ADDR` (default: this host's outbound address, that port).
async fn webrtc_from_env() -> anyhow::Result<Option<crate::webrtc::WebRtc>> {
    if std::env::var("CHIRP_RTC_PORT").as_deref() == Ok("off") {
        return Ok(None);
    }
    let port = env_or("CHIRP_RTC_PORT", 7862u16);
    let public = match std::env::var("CHIRP_PUBLIC_ADDR") {
        Ok(address) => address.parse()?,
        Err(_) => SocketAddr::new(outbound_ip().await, port),
    };
    Ok(Some(
        crate::webrtc::WebRtc::start(([0, 0, 0, 0], port).into(), Some(public)).await?,
    ))
}

/// The address this host sends from (no packet is sent), else loopback.
async fn outbound_ip() -> std::net::IpAddr {
    let probe = async {
        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
        socket.connect("8.8.8.8:80").await?;
        Ok::<_, std::io::Error>(socket.local_addr()?.ip())
    };
    probe.await.unwrap_or([127, 0, 0, 1].into())
}

/// Serves `router` on `PORT` (default 7861) until SIGTERM.
pub async fn listen(router: Router) -> anyhow::Result<()> {
    #[cfg(feature = "heap-stats")]
    let router = router.route(
        "/debug/heap",
        get(|| async {
            format!(
                "{} {}",
                crate::heap::live_bytes(),
                crate::heap::allocation_calls()
            )
        }),
    );
    let address: SocketAddr = ([0, 0, 0, 0], env_or("PORT", 7861u16)).into();
    tracing::info!(%address, "chirp listening");
    // On SIGTERM, stop accepting calls and return, so profilers can write out.
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    let listener = nodelay(tokio::net::TcpListener::bind(address).await?);
    axum::serve(listener, router)
        .with_graceful_shutdown(terminate)
        .await?;
    Ok(())
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
