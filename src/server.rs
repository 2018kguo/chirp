//! The HTTP surface: `GET /health` and `GET /v1/calls` (WebSocket upgrade).

use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use crate::brain::Brain;
use crate::call::{self, Shared};

/// Serves calls, making one brain per call with `make_brain`.
pub fn router<B, F>(shared: Shared, make_brain: F) -> Router
where
    B: Brain,
    F: Fn() -> B + Clone + Send + Sync + 'static,
{
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
                            return (axum::http::StatusCode::SERVICE_UNAVAILABLE, "at capacity")
                                .into_response();
                        }
                        // Callers send 640-byte frames: small buffers, and a cap
                        // so no one can make the server hold a huge message.
                        upgrade
                            .read_buffer_size(4 * 1024)
                            .write_buffer_size(0)
                            .max_message_size(64 * 1024)
                            .on_upgrade(move |socket| call::run(socket, shared, brain))
                            .into_response()
                    }
                },
            ),
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
