"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# the publisher's longest sequence (infer's max_length), which the engine refuses to exceed
MAX_CONTEXT = 131072
# a decoder token's caches: 12 layers of bf16 keys and values (1280 wide each) and the router's fp32 index keys (128)
CACHE_BYTES_A_TOKEN = 12 * (2 * 1280 * 2 + 128 * 4)


@dataclass
class Settings:
    model_repo: str = ""
    model_revision: str = ""
    served_name: str = ""
    model_dir: str = ""
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    zig_cpu: str = "native"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 8896
    tf_cache: str = "~/.cache/unlimited-ocr-2"
    context: int = MAX_CONTEXT
    max_tokens: int = MAX_CONTEXT
    max_pages: int = 64
    page_tokens: int = 0
    keep_warm: int = 86400
    api_key: str = ""
    extra_args: str = ""

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def model(self) -> Path:
        return self.path(self.model_dir)

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def engine_tree(self) -> Path:
        """The engine checkout the binary was built from: the pinned worktree, or the one `binary` sits in."""
        return self.run_binary.parents[3] if self.binary else self.worktree

    @property
    def chat_template(self) -> Path:
        """The family's chat template (the release ships none): every image a page, then the instruction."""
        return self.engine_tree / "zig" / "src" / "families" / "uocr" / "chat_template.jinja"

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", "--host", self.host, "--port", str(self.port), "--name", self.served_name,
             "--chat-template", str(self.chat_template), "--vision", "--vision-max-images", str(self.max_pages),
             "--context", str(self.context), "--max-tokens", str(self.max_tokens), "--keep-warm", str(self.keep_warm),
             "--no-update-check"]
        if self.page_tokens > 0:
            a += ["--vision-image-tokens", str(self.page_tokens)]
        if self.api_key:
            a += ["--api-key", self.api_key]
        a += self.extra_args.split()
        a.append(str(self.model))
        return a


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    if not (s.model / "config.json").exists():
        out.append(f"no checkpoint at {s.model}: hf download {s.model_repo} --revision {s.model_revision} --local-dir {s.model_dir}")
    if s.context < 1 or s.context > MAX_CONTEXT:
        out.append(f"context must be 1 to {MAX_CONTEXT} (the publisher's max_length)")
    if s.max_pages < 1:
        out.append("max_pages must be at least 1")
    if not s.run_binary.exists():
        out.append(f"no engine binary at {s.run_binary}: python3 -m mac build")
    elif not s.chat_template.exists():
        out.append(f"no chat template at {s.chat_template}: the engine tree lacks the unlimited-ocr family")
    return out
