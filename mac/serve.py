"""The server: started in the background (pid file, log) or in the foreground (the launchd unit), stopped, watched."""

import base64
import json
import os
import signal
import subprocess
import time
import urllib.request

from .config import CACHE_BYTES_A_TOKEN, Settings


def weights_gib(s: Settings) -> float:
    """The checkpoint's resident share once loaded: its safetensors shards."""
    return sum(f.stat().st_size for f in s.model.glob("*.safetensors")) / 2**30


def need_gib(s: Settings) -> float:
    """What a load holds: the weights, the caches for `context` tokens, and about 2 GiB of prompt and tower scratch."""
    return weights_gib(s) + s.context * CACHE_BYTES_A_TOKEN / 2**30 + 2


def health(s: Settings, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(s.url.removesuffix("/v1") + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def pid(s: Settings) -> int | None:
    """The running server's pid; a pid file whose process is gone or a zombie (exited, unreaped) is removed."""
    try:
        p = int(s.pid_file.read_text())
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout.strip()
        if state and not state.startswith("Z"):
            return p
    except Exception:
        return None
    s.pid_file.unlink(missing_ok=True)
    return None


def free_gib() -> float:
    """Memory a load can take: RAM less wired, compressed and anonymous pages (the file cache gives way)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    size = int(out.split("page size of ")[1].split()[0])
    pages = {line.split(":")[0]: int(line.split(":")[1].strip().rstrip(".")) for line in out.splitlines()[1:] if ":" in line}
    held = sum(pages.get(k, 0) for k in ("Pages wired down", "Pages occupied by compressor", "Anonymous pages"))
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    return (ram - held * size) / 2**30


def memory_check(s: Settings) -> None:
    """The weights, the caches at `context` and the scratch must be free."""
    need = need_gib(s)
    have = free_gib()
    if have < need:
        raise SystemExit(f"{have:.0f} GiB free, the server needs {need:.0f}: stop other models first")


def wait_ready(s: Settings, proc: subprocess.Popen, limit: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < limit:
        if health(s):
            return
        if proc.poll() is not None:  # exited (a refused setting, say): poll() reaps it, where kill(pid, 0) sees a zombie
            s.pid_file.unlink(missing_ok=True)
            raise SystemExit(f"the server exited while loading (status {proc.returncode}); see {s.log_file}")
        time.sleep(3)
    raise SystemExit(f"not ready after {limit:.0f} s; see {s.log_file}")


def up(s: Settings, foreground: bool = False) -> None:
    if pid(s) or health(s):
        raise SystemExit(f"already serving at {s.url}")
    memory_check(s)
    args = s.serve_args()
    if foreground:  # the launchd unit: this process becomes the server
        os.execve(args[0], args, dict(os.environ))
    s.cache.mkdir(parents=True, exist_ok=True)
    log = open(s.log_file, "ab")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    s.pid_file.write_text(str(p.pid))
    wait_ready(s, p)
    print(f"serving {s.served_name} at {s.url} (pid {p.pid})")


def down(s: Settings, grace: float = 30) -> None:
    p = pid(s)
    if not p:
        print("not running")
        return
    os.kill(p, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < grace:
        try:
            os.kill(p, 0)
        except OSError:
            break
        time.sleep(0.5)
    else:
        os.kill(p, signal.SIGKILL)
    s.pid_file.unlink(missing_ok=True)
    print("stopped")


def status(s: Settings) -> str:
    h = health(s)
    return f"pid: {pid(s) or 'not running'}\nurl: {s.url}\nhealth: {json.dumps(h) if h else 'down'}"


def ocr(s: Settings, pages: list[str], instruction: str = "", max_tokens: int = 0) -> str:
    """Page images (PNG, JPEG or WebP files, in order) to the running server; the reply's text."""
    content = []
    for p in pages:
        data = open(p, "rb").read()
        kind = "png" if data[:4] == b"\x89PNG" else "jpeg" if data[:2] == b"\xff\xd8" else "webp"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{kind};base64," + base64.b64encode(data).decode()}})
    if instruction:
        content.append({"type": "text", "text": instruction})
    body = {"model": s.served_name, "messages": [{"role": "user", "content": content}]}
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(s.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]["content"]
