//! Minimal client for talking to a running `vqtrs-api` over its Unix socket.
//!
//! Lets the CLI reuse a warm daemon (model already loaded) instead of loading a
//! model in-process on every invocation. A blocking HTTP/1.1 request with
//! `Connection: close` keeps the CLI fully synchronous and dependency-light.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};

/// Default local socket path: `$XDG_RUNTIME_DIR/vqtrs.sock`, else the temp dir.
/// Honours `VQTRS_SOCKET` as an explicit override.
fn socket_path() -> PathBuf {
    if let Some(explicit) = std::env::var_os("VQTRS_SOCKET") {
        return PathBuf::from(explicit);
    }
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join("vqtrs.sock")
}

/// Return the socket path if a daemon is currently accepting connections.
#[must_use]
pub fn live() -> Option<PathBuf> {
    let path = socket_path();
    UnixStream::connect(&path).ok().map(|_| path)
}

/// POST a JSON body to `route` over the socket and return the response body.
///
/// # Errors
///
/// Returns an error if the connection fails, the response is malformed, or the
/// server returns a non-200 status.
pub fn post_json(socket: &Path, route: &str, body: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connecting to {}", socket.display()))?;
    stream.set_read_timeout(Some(Duration::from_mins(5))).ok();

    let request = format!(
        "POST {route} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    );
    stream
        .write_all(request.as_bytes())
        .context("writing request to daemon")?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .context("reading daemon response")?;
    let text = String::from_utf8_lossy(&raw);

    let (head, payload) = text
        .split_once("\r\n\r\n")
        .context("malformed HTTP response from daemon")?;
    let status_line = head.lines().next().unwrap_or_default();
    let code = status_line.split_whitespace().nth(1).unwrap_or_default();
    if code != "200" {
        bail!("daemon returned status `{code}`: {}", payload.trim());
    }
    Ok(payload.to_owned())
}
