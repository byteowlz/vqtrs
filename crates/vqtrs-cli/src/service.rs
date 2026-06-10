//! Daemon lifecycle management for the `vqtrs-api` server.
//!
//! Ad-hoc lifecycle (`start`/`run`/`stop`/`restart`/`status`) is handled with a
//! PID file under the XDG state directory; the CLI spawns `vqtrs-api` detached
//! and records its PID, so the server itself stays free of any service code.
//! Boot persistence (`enable`/`disable`) is delegated to a systemd user unit.

use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sysinfo::{Pid, System};

/// systemd user unit name used by `enable`/`disable` (Linux only).
#[cfg(target_os = "linux")]
const UNIT_NAME: &str = "vqtrs.service";

/// Resolved running state of the daemon.
#[derive(Debug)]
pub enum Status {
    /// Running with the given PID and recorded port.
    Running {
        /// Process id of the running server.
        pid: u32,
        /// Port recorded at start time, if known.
        port: Option<u16>,
    },
    /// No PID file: the daemon is not running.
    Stopped,
    /// PID file exists but the process is gone (stale).
    Dead {
        /// The stale PID that no longer maps to a process.
        pid: u32,
    },
}

/// Manages the `vqtrs-api` daemon via a PID file under the state directory.
#[derive(Debug)]
pub struct ServiceManager {
    state_dir: PathBuf,
}

impl ServiceManager {
    /// Create a manager, ensuring the state directory exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the state directory cannot be determined or created.
    pub fn new() -> Result<Self> {
        let state_dir = state_dir()?;
        std::fs::create_dir_all(&state_dir)
            .with_context(|| format!("creating state dir {}", state_dir.display()))?;
        Ok(Self { state_dir })
    }

    fn pid_file(&self) -> PathBuf {
        self.state_dir.join("vqtrs-api.pid")
    }

    fn port_file(&self) -> PathBuf {
        self.state_dir.join("vqtrs-api.port")
    }

    /// Path to the server's redirected log file.
    #[must_use]
    pub fn log_file(&self) -> PathBuf {
        self.state_dir.join("vqtrs-api.log")
    }

    fn read_pid(&self) -> Option<u32> {
        std::fs::read_to_string(self.pid_file())
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    fn read_port(&self) -> Option<u16> {
        std::fs::read_to_string(self.port_file())
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    /// Whether a live server process is currently recorded.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.read_pid().is_some_and(process_exists)
    }

    /// Remove a PID/port file that points at a dead process.
    fn clean_stale(&self) {
        if let Some(pid) = self.read_pid()
            && !process_exists(pid)
        {
            let _ = std::fs::remove_file(self.pid_file());
            let _ = std::fs::remove_file(self.port_file());
        }
    }

    /// Spawn `vqtrs-api` detached, recording its PID and port.
    ///
    /// `api_args` is forwarded verbatim to the server; `probe_port` is the port
    /// to wait for readiness on. Returns once the port accepts connections or,
    /// if the model is still loading, once the process is confirmed alive.
    ///
    /// # Errors
    ///
    /// Returns an error if a server is already running, the binary cannot be
    /// found or spawned, or the process exits during startup.
    pub fn start(&self, api_args: &[String], probe_port: u16) -> Result<StartOutcome> {
        self.clean_stale();
        if self.is_running() {
            let pid = self.read_pid().unwrap_or(0);
            bail!("server already running (pid {pid})");
        }

        let bin = api_binary()?;
        let log = self.log_file();
        let out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .with_context(|| format!("opening log file {}", log.display()))?;
        let err = out.try_clone().context("cloning log handle")?;

        let mut cmd = Command::new(&bin);
        cmd.args(api_args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let child = cmd
            .spawn()
            .with_context(|| format!("spawning {}", bin.display()))?;
        let pid = child.id();
        self.write_pid(pid)?;
        self.write_port(probe_port)?;

        // Brief grace period to catch an immediate crash (e.g. bad model name).
        std::thread::sleep(Duration::from_millis(500));
        if !process_exists(pid) {
            let _ = std::fs::remove_file(self.pid_file());
            let _ = std::fs::remove_file(self.port_file());
            bail!("server exited immediately; see {}", log.display());
        }

        if wait_for_port(probe_port, 100) {
            Ok(StartOutcome::Ready {
                pid,
                port: probe_port,
            })
        } else if process_exists(pid) {
            Ok(StartOutcome::Initializing { pid })
        } else {
            let _ = std::fs::remove_file(self.pid_file());
            let _ = std::fs::remove_file(self.port_file());
            bail!("server exited during startup; see {}", log.display());
        }
    }

    /// Stop the running server and remove its PID/port files.
    ///
    /// # Errors
    ///
    /// Returns an error if no server is running or the kill command fails.
    pub fn stop(&self) -> Result<()> {
        let Some(pid) = self.read_pid() else {
            bail!("server not running");
        };
        if !process_exists(pid) {
            let _ = std::fs::remove_file(self.pid_file());
            let _ = std::fs::remove_file(self.port_file());
            bail!("server not running (cleaned up stale pid file)");
        }

        kill(pid)?;
        for _ in 0..50 {
            if !process_exists(pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = std::fs::remove_file(self.pid_file());
        let _ = std::fs::remove_file(self.port_file());
        Ok(())
    }

    /// Stop (if running) then start the server.
    ///
    /// # Errors
    ///
    /// Returns an error if stopping or starting fails.
    pub fn restart(&self, api_args: &[String], probe_port: u16) -> Result<StartOutcome> {
        if self.is_running() {
            self.stop()?;
            std::thread::sleep(Duration::from_millis(200));
        }
        self.start(api_args, probe_port)
    }

    /// Current running state of the daemon.
    #[must_use]
    pub fn status(&self) -> Status {
        match self.read_pid() {
            Some(pid) if process_exists(pid) => Status::Running {
                pid,
                port: self.read_port(),
            },
            Some(pid) => Status::Dead { pid },
            None => Status::Stopped,
        }
    }

    fn write_pid(&self, pid: u32) -> Result<()> {
        std::fs::write(self.pid_file(), pid.to_string()).context("writing pid file")
    }

    fn write_port(&self, port: u16) -> Result<()> {
        std::fs::write(self.port_file(), port.to_string()).context("writing port file")
    }
}

/// Result of a successful [`ServiceManager::start`].
#[derive(Debug)]
pub enum StartOutcome {
    /// The server is accepting connections on `port`.
    Ready {
        /// Process id of the server.
        pid: u32,
        /// Port the server is listening on.
        port: u16,
    },
    /// The process is alive but the port is not open yet (model still loading).
    Initializing {
        /// Process id of the server.
        pid: u32,
    },
}

/// Run `vqtrs-api` in the foreground (blocking), inheriting stdio.
///
/// # Errors
///
/// Returns an error if the binary cannot be found or the process fails.
pub fn run(api_args: &[String]) -> Result<()> {
    let bin = api_binary()?;
    let status = Command::new(&bin)
        .args(api_args)
        .status()
        .with_context(|| format!("running {}", bin.display()))?;
    if !status.success() {
        bail!("server exited with {status}");
    }
    Ok(())
}

/// Build the `vqtrs-api` argument list from optional overrides.
#[must_use]
pub fn api_args(
    host: &str,
    port: u16,
    model: Option<&str>,
    rerank_model: Option<&str>,
    sparse_model: Option<&str>,
    m3_model: Option<&str>,
) -> Vec<String> {
    let mut args = vec![
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
    ];
    for (flag, value) in [
        ("--model", model),
        ("--rerank-model", rerank_model),
        ("--sparse-model", sparse_model),
        ("--m3-model", m3_model),
    ] {
        if let Some(v) = value {
            args.push(flag.to_owned());
            args.push(v.to_owned());
        }
    }
    args
}

/// Resolve the `vqtrs-api` binary: next to this executable, else from `PATH`.
fn api_binary() -> Result<PathBuf> {
    let name = format!("vqtrs-api{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().context("locating current executable")?;
    if let Some(dir) = exe.parent() {
        let sibling = dir.join(&name);
        if sibling.exists() {
            return Ok(sibling);
        }
    }
    Ok(PathBuf::from(name))
}

fn process_exists(pid: u32) -> bool {
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, false);
    sys.process(Pid::from_u32(pid)).is_some()
}

/// Poll a localhost TCP port for readiness, up to `attempts` × 100ms.
fn wait_for_port(port: u16, attempts: u32) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    for _ in 0..attempts {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

#[cfg(unix)]
fn kill(pid: u32) -> Result<()> {
    let status = Command::new("kill")
        .arg(pid.to_string())
        .status()
        .context("running kill")?;
    if !status.success() {
        bail!("failed to signal pid {pid}");
    }
    Ok(())
}

#[cfg(windows)]
fn kill(pid: u32) -> Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .status()
        .context("running taskkill")?;
    if !status.success() {
        bail!("failed to kill pid {pid}");
    }
    Ok(())
}

fn state_dir() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("state")))
        .context("could not determine state directory")?;
    Ok(base.join("vqtrs"))
}

// ----------------------------------------------------------------------------
// systemd user unit (enable / disable / status integration), Linux only.
// ----------------------------------------------------------------------------

/// systemd enabled/active state, when a unit exists.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct SystemdState {
    /// Output of `systemctl --user is-enabled` (e.g. `enabled`, `disabled`).
    pub enabled: String,
    /// Output of `systemctl --user is-active` (e.g. `active`, `inactive`).
    pub active: String,
}

/// Install and enable a systemd user unit that runs `vqtrs-api`.
///
/// # Errors
///
/// Returns an error if the unit cannot be written or `systemctl` fails.
#[cfg(target_os = "linux")]
pub fn enable(api_args: &[String]) -> Result<PathBuf> {
    let bin = api_binary()?;
    let unit_path = unit_path()?;
    if let Some(parent) = unit_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let exec = if api_args.is_empty() {
        bin.display().to_string()
    } else {
        format!("{} {}", bin.display(), api_args.join(" "))
    };
    let unit = format!(
        "[Unit]\n\
         Description=vqtrs embeddings + reranking server\n\
         After=network.target\n\n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exec}\n\
         Restart=on-failure\n\
         RestartSec=5\n\n\
         [Install]\n\
         WantedBy=default.target\n"
    );
    std::fs::write(&unit_path, unit).with_context(|| format!("writing {}", unit_path.display()))?;

    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", "--now", UNIT_NAME])?;
    Ok(unit_path)
}

/// Disable and remove the systemd user unit.
///
/// # Errors
///
/// Returns an error if `systemctl` fails.
#[cfg(target_os = "linux")]
pub fn disable() -> Result<()> {
    systemctl(&["disable", "--now", UNIT_NAME])?;
    if let Ok(path) = unit_path() {
        let _ = std::fs::remove_file(path);
    }
    systemctl(&["daemon-reload"])?;
    Ok(())
}

/// Query the systemd user unit's enabled/active state, if it is installed.
#[cfg(target_os = "linux")]
#[must_use]
pub fn systemd_state() -> Option<SystemdState> {
    let unit = unit_path().ok()?;
    if !unit.exists() {
        return None;
    }
    Some(SystemdState {
        enabled: systemctl_output(&["is-enabled", UNIT_NAME])
            .unwrap_or_else(|| "unknown".to_owned()),
        active: systemctl_output(&["is-active", UNIT_NAME]).unwrap_or_else(|| "unknown".to_owned()),
    })
}

#[cfg(target_os = "linux")]
fn unit_path() -> Result<PathBuf> {
    let config = dirs::config_dir().context("could not determine config directory")?;
    Ok(config.join("systemd").join("user").join(UNIT_NAME))
}

#[cfg(target_os = "linux")]
fn systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("running systemctl --user")?;
    if !status.success() {
        bail!("`systemctl --user {}` failed", args.join(" "));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn systemctl_output(args: &[&str]) -> Option<String> {
    let output = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if text.is_empty() { None } else { Some(text) }
}

/// `enable` is only supported on Linux (systemd).
#[cfg(not(target_os = "linux"))]
pub fn enable(_api_args: &[String]) -> Result<PathBuf> {
    bail!("`server enable` requires systemd (Linux only)");
}

/// `disable` is only supported on Linux (systemd).
#[cfg(not(target_os = "linux"))]
pub fn disable() -> Result<()> {
    bail!("`server disable` requires systemd (Linux only)");
}
