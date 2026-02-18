// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::Path;
use std::process::Command;

fn set_env(name: &str, value: Option<String>) {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            println!("cargo:rustc-env={name}={trimmed}");
        }
    }
}

fn command_output(program: &str, args: &[&str], cwd: Option<&Path>) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn command_stdout_bytes(program: &str, args: &[&str], cwd: Option<&Path>) -> Option<Vec<u8>> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(output.stdout)
}

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=PROFILE");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok();
    let manifest_path = manifest_dir.as_deref().map(Path::new);

    if let Some(manifest_path) = manifest_path {
        let git_head = manifest_path.join("../../.git/HEAD");
        if git_head.exists() {
            println!("cargo:rerun-if-changed={}", git_head.display());
        }
    }

    set_env("CPD_BUILD_TARGET_TRIPLE", env::var("TARGET").ok());
    set_env("CPD_BUILD_PROFILE", env::var("PROFILE").ok());

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    set_env(
        "CPD_BUILD_RUSTC_VERSION",
        command_output(rustc.as_str(), &["-V"], manifest_path),
    );

    set_env(
        "CPD_BUILD_GIT_SHA",
        command_output("git", &["rev-parse", "--short=12", "HEAD"], manifest_path),
    );

    let git_dirty = command_stdout_bytes(
        "git",
        &["status", "--porcelain", "--untracked-files=no"],
        manifest_path,
    )
    .map(|stdout| (!stdout.is_empty()).to_string());
    set_env("CPD_BUILD_GIT_DIRTY", git_dirty);
}
