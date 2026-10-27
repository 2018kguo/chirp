//! Downloads the pinned models into `models/`, where the tests and benches
//! read them (`chirp::server::serve` fetches its own).

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    chirp::models::ensure(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/models"
    )))
    .await
}
