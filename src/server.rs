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
                        upgrade
                            .on_upgrade(move |socket| call::run(socket, shared, brain))
                            .into_response()
                    }
                },
            ),
        )
        .with_state(shared)
}
