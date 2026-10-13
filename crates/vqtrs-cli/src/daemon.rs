//! Minimal client for talking to a running `vqtrs-api` over its Unix socket.
//!
//! Lets the CLI reuse a warm daemon (model already loaded) instead of loading a
//! model in-process on every invocation. A blocking HTTP/1.1 request with
//! `Connection: close` keeps the CLI fully synchronous and dependency-light.

use std::ffi::OsStr;
#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::time::Duration;

#[cfg(unix)]
use anyhow::Context;
use anyhow::{Result, bail};

/// Reject an explicit socket override on platforms without Unix socket support.
///
/// # Errors
///
/// Returns an error when a socket was explicitly requested on an unsupported platform.
pub fn validate_override(supported: bool, explicit: Option<&OsStr>) -> Result<()> {
    if !supported && explicit.is_some() {
        bail!(
            "Unix sockets are unavailable on this platform; unset VQTRS_SOCKET and use local CLI inference or the HTTP API"
        );
    }
    Ok(())
}

/// Default local socket path: `$XDG_RUNTIME_DIR/vqtrs.sock`, else the temp dir.
/// Honours `VQTRS_SOCKET` as an explicit override.
#[cfg(unix)]
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
#[cfg(unix)]
pub fn live() -> Option<PathBuf> {
    let path = socket_path();
    UnixStream::connect(&path).ok().map(|_| path)
}

/// Non-Unix clients use in-process inference; no implicit TCP daemon discovery.
#[cfg(not(unix))]
#[must_use]
pub const fn live() -> Option<PathBuf> {
    None
}

/// POST a JSON body to `route` over the socket and return the response body.
///
/// # Errors
///
/// Returns an error if the connection fails, the response is malformed, or the
/// server returns a non-200 status.
#[cfg(unix)]
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

/// Unix socket requests are unsupported on this platform.
///
/// # Errors
///
/// Always returns an explicit unsupported-platform error, never a TCP fallback.
#[cfg(not(unix))]
pub fn post_json(_socket: &Path, _route: &str, _body: &str) -> Result<String> {
    bail!("Unix socket requests are unavailable on this platform; use the HTTP API");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_policy_is_explicit_without_mutating_environment() {
        let explicit = Some(OsStr::new("chosen.sock"));
        assert!(validate_override(true, explicit).is_ok());
        assert!(validate_override(false, None).is_ok());
        assert!(validate_override(false, explicit).is_err());
    }

    #[cfg(not(unix))]
    #[test]
    fn non_unix_has_no_warm_socket_and_rejects_post() {
        assert!(live().is_none());
        assert!(post_json(Path::new("unused.sock"), "/v1/embeddings", "{}").is_err());
    }
}
