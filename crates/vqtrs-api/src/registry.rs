//! A keyed, keep-warm model registry with LRU eviction.
//!
//! Models load lazily on first request for a given name and stay warm. Models
//! preloaded via [`Registry::load_pinned`] (the configured "warm" set) are never
//! evicted; the rest are evicted least-recently-used once `max` is exceeded
//! (`max == 0` means unbounded). Different models live behind independent
//! `Arc`s, so inference on distinct models runs in parallel.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Result, anyhow};

/// A bounded cache of loaded models keyed by name.
#[derive(Debug)]
pub struct Registry<T> {
    inner: Mutex<Inner<T>>,
    max: usize,
}

#[derive(Debug)]
struct Inner<T> {
    models: HashMap<String, Arc<T>>,
    lru: Vec<String>,
    pinned: HashSet<String>,
}

impl<T> Registry<T> {
    /// Create a registry holding at most `max` models (`0` = unbounded).
    #[must_use]
    pub fn new(max: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                models: HashMap::new(),
                lru: Vec::new(),
                pinned: HashSet::new(),
            }),
            max,
        }
    }

    /// Return the model named `key`, loading it (outside the lock) on a miss.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned or `load` fails.
    pub fn get_or_load(&self, key: &str, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        if let Some(found) = self.touch_existing(key)? {
            return Ok(found);
        }
        let model = Arc::new(load()?);
        self.insert(key, model, false)
    }

    /// Load `key` and pin it so it is never evicted (used for warm models).
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned or `load` fails.
    pub fn load_pinned(&self, key: &str, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        if let Some(found) = self.touch_existing(key)? {
            self.lock()?.pinned.insert(key.to_owned());
            return Ok(found);
        }
        let model = Arc::new(load()?);
        self.insert(key, model, true)
    }

    fn touch_existing(&self, key: &str) -> Result<Option<Arc<T>>> {
        let mut guard = self.lock()?;
        let model = guard.models.get(key).cloned();
        if model.is_some() {
            move_to_back(&mut guard.lru, key);
        }
        drop(guard);
        Ok(model)
    }

    fn insert(&self, key: &str, model: Arc<T>, pin: bool) -> Result<Arc<T>> {
        let mut guard = self.lock()?;
        // A concurrent request may have loaded the same model meanwhile.
        if let Some(existing) = guard.models.get(key).cloned() {
            move_to_back(&mut guard.lru, key);
            if pin {
                guard.pinned.insert(key.to_owned());
            }
            drop(guard);
            return Ok(existing);
        }
        guard.models.insert(key.to_owned(), Arc::clone(&model));
        guard.lru.push(key.to_owned());
        if pin {
            guard.pinned.insert(key.to_owned());
        }
        self.evict(&mut guard);
        drop(guard);
        Ok(model)
    }

    fn evict(&self, guard: &mut Inner<T>) {
        if self.max == 0 {
            return;
        }
        while guard.models.len() > self.max {
            let victim = guard
                .lru
                .iter()
                .find(|k| !guard.pinned.contains(*k))
                .cloned();
            match victim {
                Some(key) => {
                    guard.lru.retain(|k| k != &key);
                    guard.models.remove(&key);
                }
                None => break, // everything left is pinned
            }
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner<T>>> {
        self.inner
            .lock()
            .map_err(|_| anyhow!("model registry lock poisoned"))
    }
}

fn move_to_back(lru: &mut Vec<String>, key: &str) {
    if let Some(pos) = lru.iter().position(|k| k == key) {
        let key = lru.remove(pos);
        lru.push(key);
    }
}
