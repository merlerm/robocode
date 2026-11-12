"""Drive the real CLI with deterministic local API responses; no inference."""

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

fixed = (
    json.loads(Path("/sandbox/.claude/settings.json").read_text(encoding="utf-8"))
    .get("env", {})
    .get("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS")
    == "1"
)
root = Path("/sandbox")
responses = []
final_sent = threading.Event()


class Handler(BaseHTTPRequestHandler):
    """Serve the scripted Bash calls and a final text response."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        """Return the next tool call based on received tool results."""
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "count_tokens" in self.path:
            payload = json.dumps({"input_tokens": 10}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)
            return
        if "/messages" not in self.path:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            return
        results = [
            b
            for m in data.get("messages", [])
            for b in (
                m.get("content", []) if isinstance(m.get("content"), list) else []
            )
            if b.get("type") == "tool_result"
        ]
        if not results:
            block = {
                "type": "tool_use",
                "id": "toolu_timeout_probe",
                "name": "Bash",
                "input": {
                    "command": "echo $$ > wait.pid; while :; do sleep 1; done",
                    "timeout": 1000,
                    "description": "Exercise a stuck monitoring shell",
                },
            }
            stop = "tool_use"
        elif len(results) == 1:
            responses.append(results[-1])
            block = {
                "type": "tool_use",
                "id": "toolu_parallel_probe",
                "name": "Bash",
                "input": {
                    "command": (
                        "sleep 0.05 & a=$!; sleep 0.05 & b=$!; "
                        "wait $a $b; echo PARALLEL_OK"
                    ),
                    "timeout": 1000,
                    "description": "Check parallel work with explicit wait",
                },
            }
            stop = "tool_use"
        else:
            responses.append(results[-1])
            block = {"type": "text", "text": "Fixture finished."}
            stop = "end_turn"
        msg = {
            "id": "msg_probe",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [block],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        self.send_response(200)
        if data.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {**msg, "content": [], "stop_reason": None},
                    },
                )
            ]
            empty = (
                {"type": "text", "text": ""}
                if block["type"] == "text"
                else {**block, "input": {}}
            )
            delta = (
                {"type": "text_delta", "text": block["text"]}
                if block["type"] == "text"
                else {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                }
            )
            events += [
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": empty},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": None},
                        "usage": {"output_tokens": 10},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, event in events:
                self.wfile.write(
                    f"event: {name}\ndata: {json.dumps(event)}\n\n".encode()
                )
        else:
            self.end_headers()
            self.wfile.write(json.dumps(msg).encode())
        self.wfile.flush()
        if stop == "end_turn":
            final_sent.set()


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
env = {
    **os.environ,
    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    "ANTHROPIC_AUTH_TOKEN": "fixture-only",
    "CLAUDE_CONFIG_DIR": "/sandbox/cli-config",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "BASH_DEFAULT_TIMEOUT_MS": "1000",
    "BASH_MAX_TIMEOUT_MS": "1000",
}
cmd = [
    "claude",
    "-p",
    "--model",
    "claude-opus-5-5",
    "--output-format",
    "stream-json",
    "--verbose",
    "--dangerously-skip-permissions",
    "--tools",
    "Bash",
    "--strict-mcp-config",
    "--setting-sources",
    "project",
]
start = time.monotonic()
with (
    open("/sandbox/stream.jsonl", "w", encoding="utf-8") as out,
    open("/sandbox/stderr.log", "w", encoding="utf-8") as err,
):
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=out,
        stderr=err,
        text=True,
        start_new_session=True,
    )
    proc.stdin.write("Execute the scripted fixture.")
    proc.stdin.close()
    try:
        assert final_sent.wait(25), (
            "No final fixture response: "
            + Path("/sandbox/stderr.log").read_text(encoding="utf-8")[-2000:]
        )
        try:
            proc.wait(timeout=5)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        pid = int((root / "wait.pid").read_text(encoding="utf-8"))
        alive = Path(f"/proc/{pid}").exists()
        summary = {
            "fixed": fixed,
            "cli_exited": exited,
            "wait_shell_alive": alive,
            "seconds": time.monotonic() - start,
            "tool_results": responses,
            "cli_exit": proc.poll(),
        }
        print(json.dumps(summary), flush=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        assert exited == fixed and alive != fixed, summary
        assert "PARALLEL_OK" in str(responses[-1]), responses
        if fixed:
            assert responses[0]["is_error"] and "timed out" in str(
                responses[0]
            ), responses
            native_events = [
                json.loads(line)
                for line in (root / "stream.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.startswith("{")
            ]
            assert any(
                e.get("type") == "result" and not e.get("is_error")
                for e in native_events
            ), native_events
        else:
            assert "moved to the background" in str(responses[0]), responses
    finally:
        # The control's shell may have its own process group; stop it explicitly.
        wait_pid = root / "wait.pid"
        if wait_pid.exists():
            try:
                os.kill(int(wait_pid.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        server.shutdown()
