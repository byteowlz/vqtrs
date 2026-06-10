# vqtrs justfile (Rust workspace)

# Default recipe - show help
default:
    @just --list

# === Development ===

# Install both binaries (vqtrs + vqtrs-api), CPU / ONNX only
install:
    cargo install --path crates/vqtrs-cli --force
    cargo install --path crates/vqtrs-api --force

# Interactive install with hardware acceleration selection
install-all:
    ./scripts/install-vqtrs.sh

# Install with the Qwen3 candle backend (CPU)
install-qwen3:
    cargo install --path crates/vqtrs-cli --features qwen3 --force
    cargo install --path crates/vqtrs-api --features qwen3 --force

# Install for NVIDIA: ONNX CUDA EP + Qwen3 on CUDA
install-cuda:
    cargo install --path crates/vqtrs-cli --features cuda,qwen3-cuda --force
    cargo install --path crates/vqtrs-api --features cuda,qwen3-cuda --force

# Install for Apple Silicon: ONNX CoreML + Qwen3 on Metal
install-mac:
    cargo install --path crates/vqtrs-cli --features coreml,qwen3-metal --force
    cargo install --path crates/vqtrs-api --features coreml,qwen3-metal --force

# Uninstall all binaries
uninstall:
    @for crate in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "bin") | .name'); do \
        echo "Uninstalling $crate..."; \
        cargo uninstall $crate 2>/dev/null || true; \
    done

# === Building ===

# Debug build (all crates)
build:
    cargo build --workspace

# Release build (all crates)
build-release:
    cargo build --workspace --release

# Build a specific crate
build-crate CRATE:
    cargo build -p {{CRATE}}

# Build with all features
build-all:
    cargo build --workspace --all-features

# Fast compile check
check:
    cargo check --workspace

# Check a specific crate
check-crate CRATE:
    cargo check -p {{CRATE}}

# Clean build artifacts
clean:
    cargo clean

# === Testing ===

# Run all tests
test:
    cargo test --workspace

# Run tests for a specific crate
test-crate CRATE:
    cargo test -p {{CRATE}}

# Run tests with all features
test-all:
    cargo test --workspace --all-features

# Run tests verbosely
test-v:
    cargo test --workspace -- --nocapture

# Run a specific test
test-one TEST:
    cargo test --workspace {{TEST}}

# === Code Quality ===

# Format all code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run clippy on all crates
clippy:
    cargo clippy --workspace -- -D warnings

# Alias for clippy
lint: clippy

# Run ast-grep guardrails for Rust sources
lint-rust-ai-guardrails:
    sg scan --config .ast-grep/sgconfig.yml

# Clippy on a specific crate
clippy-crate CRATE:
    cargo clippy -p {{CRATE}} -- -D warnings

# Auto-fix clippy warnings
fix:
    cargo clippy --workspace --fix --allow-dirty

# Run all checks
check-all: fmt-check clippy lint-rust-ai-guardrails test

# === Config Generation ===

# Regenerate examples/config.schema.json from the ServerConfig struct
generate-schema:
    cargo run -q -p vqtrs-api -- --print-schema > examples/config.schema.json

# === Documentation ===

# Generate docs for all crates
docs:
    cargo doc --workspace --no-deps

# Generate and open docs
docs-open:
    cargo doc --workspace --no-deps --open

# Docs for a specific crate
docs-crate CRATE:
    cargo doc -p {{CRATE}} --no-deps --open

# === Dependencies ===

# Update all dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated --workspace

# === Workspace Info ===

# List all crates in workspace
list:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | "\(.name) (\(.version))"'

# List binary crates
list-bins:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "bin") | .name'

# List library crates
list-libs:
    @cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.targets[] | .kind[] == "lib") | .name'

# === Release ===

# Release build and show binary sizes
release: build-release
    @echo "Binary sizes:"
    @find target/release -maxdepth 1 -type f -perm +111 ! -name "*.d" -exec ls -lh {} \; 2>/dev/null || true

# Tag and push a release
release-tag VERSION:
    git tag v{{VERSION}}
    git push --tags

# Set up GitHub secrets for automated releases (requires byt)
setup-secrets:
    byt secrets setup vqtrs
