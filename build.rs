//! The prebuilt ONNX Runtime's CoreML code calls `__isPlatformVersionAtLeast`,
//! which lives in clang's runtime library; rustc links with -nodefaultlibs,
//! so link that library explicitly on macOS.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let output = std::process::Command::new("clang")
        .arg("--print-resource-dir")
        .output()
        .expect("clang --print-resource-dir");
    let resource_dir = String::from_utf8(output.stdout).expect("utf-8 path");
    println!(
        "cargo:rustc-link-search=native={}/lib/darwin",
        resource_dir.trim()
    );
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
