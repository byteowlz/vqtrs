//! One owned process group and I/O thread per loaded native engine.

use std::io::{BufRead as _, Read as _, Write as _};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;

use tempfile::TempDir;

use super::prepare::Prepared;
use super::{Reply, Request, Runtime, error, fail, validate};
use crate::error::Result;

const RESPONSE_LIMIT: u64 = 512 * 1024;
const START_TIMEOUT: Duration = Duration::from_mins(15);
const INFERENCE_TIMEOUT: Duration = Duration::from_mins(5);

pub(super) struct Worker {
    child: Child,
    send: Option<SyncSender<Vec<u8>>>,
    replies: Receiver<Result<Reply>>,
    io: Option<JoinHandle<()>>,
    _directory: TempDir,
    runtime: Runtime,
    sequence: u64,
    broken: bool,
}

impl Worker {
    pub(super) fn start(runtime: Runtime, root: &Path, python: &Path) -> Result<Self> {
        let directory = tempfile::tempdir()?;
        for (name, source) in [
            ("native_worker.py", include_str!("../native_worker.py")),
            ("coreml_backend.py", include_str!("../coreml_backend.py")),
            ("mlx_backend.py", include_str!("../mlx_backend.py")),
            ("coreml-assets.json", include_str!("../coreml-assets.json")),
            ("mlx-assets.json", include_str!("../mlx-assets.json")),
            ("mlx-source.json", include_str!("../mlx-source.json")),
            ("FluidUse-LICENSE", include_str!("../FluidUse-LICENSE")),
            ("mlx-vlm-LICENSE", include_str!("../mlx-vlm-LICENSE")),
        ] {
            std::fs::write(directory.path().join(name), source)?;
        }
        let mut command = Command::new("uv");
        command
            .args(["run", "--no-project"])
            .arg(python)
            .arg("-I")
            .arg(directory.path().join("native_worker.py"))
            .arg(runtime.argument())
            .arg(root)
            .env("TOKENIZERS_PARALLELISM", "false")
            .env("OMP_NUM_THREADS", "1")
            .env("VECLIB_MAXIMUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        Self::spawn(command, directory, runtime)
    }

    fn spawn(mut command: Command, directory: TempDir, runtime: Runtime) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        let mut child = command.spawn()?;
        let pipes = child.stdin.take().zip(child.stdout.take());
        let Some((mut stdin, stdout)) = pipes else {
            terminate(&mut child);
            return Err(fail("native worker requires stdin/stdout pipes"));
        };
        let (send, requests) = mpsc::sync_channel::<Vec<u8>>(1);
        let (responses, replies) = mpsc::sync_channel(1);
        let io = std::thread::Builder::new()
            .name("vqtrs-native-io".into())
            .spawn(move || {
                let mut reader = std::io::BufReader::new(stdout);
                if responses.send(read_reply(&mut reader)).is_err() {
                    return;
                }
                while let Ok(request) = requests.recv() {
                    let reply = stdin
                        .write_all(&request)
                        .and_then(|()| stdin.flush())
                        .map_err(error)
                        .and_then(|()| read_reply(&mut reader));
                    if responses.send(reply).is_err() {
                        return;
                    }
                }
            })
            .map_err(|failure| {
                terminate(&mut child);
                error(failure)
            })?;
        let worker = Self {
            child,
            send: Some(send),
            replies,
            io: Some(io),
            _directory: directory,
            runtime,
            sequence: 0,
            broken: false,
        };
        let ready = worker.receive(START_TIMEOUT)?;
        validate(ready, runtime, None, 0)?;
        Ok(worker)
    }

    fn receive(&self, deadline: Duration) -> Result<Reply> {
        self.replies.recv_timeout(deadline).map_err(|_| {
            fail("native worker exited or exceeded its deadline; reload explicitly (no fallback)")
        })?
    }

    pub(super) fn predict(&mut self, items: &[Prepared]) -> Result<Vec<Vec<f32>>> {
        if self.broken {
            return Err(fail(
                "native worker is unavailable after an earlier failure; unload/reload explicitly",
            ));
        }
        let result = self.predict_inner(items);
        if result.is_err() {
            self.stop();
        }
        result
    }

    fn predict_inner(&mut self, items: &[Prepared]) -> Result<Vec<Vec<f32>>> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("native request sequence exhausted"))?;
        let mut bytes = serde_json::to_vec(&Request {
            protocol: 1,
            id: self.sequence,
            items: items.iter().map(|p| &p.item).collect(),
        })
        .map_err(error)?;
        if bytes.len() >= RESPONSE_LIMIT as usize {
            return Err(fail("native request metadata size limit"));
        }
        bytes.push(b'\n');
        self.send
            .as_ref()
            .ok_or_else(|| fail("native worker stopped"))?
            .send(bytes)
            .map_err(error)?;
        let reply = self.receive(INFERENCE_TIMEOUT)?;
        validate(reply, self.runtime, Some(self.sequence), items.len())
    }

    fn stop(&mut self) {
        if self.broken && self.io.is_none() {
            return;
        }
        self.broken = true;
        // uv owns the Python descendant. Kill only this engine's fresh process
        // group; never stop a shared framework compiler or unrelated service.
        terminate(&mut self.child);
        self.send.take();
        if let Some(io) = self.io.take() {
            let _ = io.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("/bin/kill")
            .args(["-KILL", &format!("-{}", child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn read_reply(reader: &mut std::io::BufReader<std::process::ChildStdout>) -> Result<Reply> {
    let mut bytes = Vec::new();
    reader
        .take(RESPONSE_LIMIT + 1)
        .read_until(b'\n', &mut bytes)?;
    if bytes.is_empty() || bytes.len() > RESPONSE_LIMIT as usize || bytes.last() != Some(&b'\n') {
        return Err(fail(
            "native worker returned a truncated/oversized response",
        ));
    }
    serde_json::from_slice(&bytes).map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn command(script: &str) -> Command {
        let mut command = Command::new("sh");
        command
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        command
    }

    #[cfg(unix)]
    #[test]
    fn worker_protocol_and_failure_cleanup() {
        let hello = format!(
            r#"printf '%s\n' '{{"protocol":1,"model":"{}"}}'; read line; printf '%s\n' '{{"protocol":1,"model":"wrong","id":1,"vectors":[]}}'; sleep 30"#,
            Runtime::CoreMl.identity()
        );
        let mut worker = Worker::spawn(
            command(&hello),
            tempfile::tempdir().unwrap(),
            Runtime::CoreMl,
        )
        .unwrap();
        assert!(worker.predict(&[]).is_err());
        assert!(worker.child.try_wait().unwrap().is_some());
        assert!(worker.predict(&[]).is_err());
        assert!(worker.io.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn invalid_frames_and_deadline_fail_closed() {
        for script in [
            "printf '{}\\n'; sleep 30",
            "printf 'truncated'; exit 0",
            "head -c 524289 /dev/zero; sleep 30",
        ] {
            assert!(
                Worker::spawn(
                    command(script),
                    tempfile::tempdir().unwrap(),
                    Runtime::CoreMl
                )
                .is_err()
            );
        }
        let hello = format!(
            r#"printf '%s\n' '{{"protocol":1,"model":"{}"}}'; sleep 30"#,
            Runtime::CoreMl.identity()
        );
        let mut worker = Worker::spawn(
            command(&hello),
            tempfile::tempdir().unwrap(),
            Runtime::CoreMl,
        )
        .unwrap();
        assert!(worker.receive(Duration::from_millis(20)).is_err());
        worker.stop();
        assert!(worker.child.try_wait().unwrap().is_some());
        assert!(worker.io.is_none());
    }

    #[cfg(unix)]
    #[test]
    #[expect(
        clippy::used_underscore_binding,
        reason = "test inspects the private lifetime guard that production only retains"
    )]
    fn drop_reaps_worker_and_removes_private_files() {
        let hello = format!(
            r#"printf '%s\n' '{{"protocol":1,"model":"{}"}}'; sleep 30"#,
            Runtime::CoreMl.identity()
        );
        let worker = Worker::spawn(
            command(&hello),
            tempfile::tempdir().unwrap(),
            Runtime::CoreMl,
        )
        .unwrap();
        let path = worker._directory.path().to_owned();
        let pid = worker.child.id();
        drop(worker);
        assert!(!path.exists());
        assert!(
            !Command::new("/bin/kill")
                .args(["-0", &pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success()
        );
    }
}
