//! Runs the scripted mocks on a port, for poking at engines by hand.
//!
//!     cargo run --release --bin mocks -- <port>

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port: u16 = std::env::args().nth(1).map_or(7870, |p| p.parse().unwrap());
    let address =
        chirp::mock::start_on(("127.0.0.1", port), chirp::mock::Script::default()).await?;
    println!("mocks on {address}");
    std::future::pending::<()>().await;
    Ok(())
}
