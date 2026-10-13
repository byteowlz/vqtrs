//! Platform-aware optional local socket; TCP remains available everywhere.

use std::path::PathBuf;

#[cfg(unix)]
use anyhow::Context;
use anyhow::{Result, bail};
use axum::Router;

/// Select a socket only on supported platforms; explicit unsupported requests fail.
///
/// # Errors
///
/// Returns an error for an explicit path on a platform without Unix sockets.
pub fn select(
    supported: bool,
    disabled: bool,
    explicit: Option<PathBuf>,
) -> Result<Option<PathBuf>> {
    if !supported && explicit.is_some() {
        bail!(
            "Unix sockets are unavailable on this platform; remove --socket/VQTRS_SOCKET/config socket and use TCP"
        );
    }
    Ok((supported && !disabled).then(|| explicit.unwrap_or_else(default_path)))
}

fn default_path() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join("vqtrs.sock")
}

/// Bind the optional Unix listener and spawn its owned server task.
///
/// # Errors
///
/// Returns an error when binding a selected socket fails.
#[cfg(unix)]
pub fn serve(app: &Router, path: Option<PathBuf>) -> Result<()> {
    if let Some(path) = path {
        let _ = std::fs::remove_file(&path);
        let listener = tokio::net::UnixListener::bind(&path)
            .with_context(|| format!("binding unix socket {}", path.display()))?;
        log::info!("serving on unix:{}", path.display());
        let app = app.clone();
        tokio::spawn(async move {
            if let Err(err) = axum::serve(listener, app).await {
                log::error!("unix socket server error: {err}");
            }
        });
    }
    Ok(())
}

/// Validate that no Unix listener was selected; TCP startup stays in the caller.
///
/// # Errors
///
/// Returns an error if an unsupported Unix listener was selected.
#[cfg(not(unix))]
pub fn serve(_app: &Router, path: Option<PathBuf>) -> Result<()> {
    if path.is_some() {
        bail!("Unix socket listener selected on an unsupported platform");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_disabled_platform_policies() {
        assert!(select(true, false, None).unwrap().is_some());
        assert!(select(true, true, None).unwrap().is_none());
        assert!(select(false, false, None).unwrap().is_none());
        assert!(select(false, true, None).unwrap().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_listener_still_serves_http() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("vqtrs-uds-{nonce}.sock"));
        let app = Router::new().route("/probe", axum::routing::get(|| async { "warm" }));
        serve(&app, Some(path.clone())).unwrap();
        let mut client = tokio::net::UnixStream::connect(&path).await.unwrap();
        client
            .write_all(b"GET /probe HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(3),
            client.read_to_end(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.ends_with("warm"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn explicit_paths_never_fall_back_on_unsupported_platforms() {
        let path = PathBuf::from("chosen.sock");
        assert_eq!(
            select(true, false, Some(path.clone())).unwrap(),
            Some(path.clone())
        );
        assert!(select(true, true, Some(path.clone())).unwrap().is_none());
        assert!(select(false, false, Some(path.clone())).is_err());
        assert!(select(false, true, Some(path)).is_err());
    }
}
