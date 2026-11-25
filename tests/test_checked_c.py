import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest import TestCase
from unittest.mock import Mock, patch

from src.checked_c import Case, request_completion, run_container, run_task, sandbox_test, solve

DRAFT = "```c\nint main(void) { return 1; }\n```"
FIX = "```c\nint main(void) { return 0; }\n```"


class CheckedCTests(TestCase):
    def test_revises_once_and_reserves_held_out_cases_for_scoring(self):
        requests = []
        public = {"stdin": "public input", "stdout": "expected output"}
        private = {"stdin": "reserved input", "stdout": "reserved output"}

        def generate(messages):
            requests.append([turn.copy() for turn in messages])
            return (DRAFT if len(requests) == 1 else FIX), "stop"

        def run_test(source, case):
            if "return 1" in source:
                return f"Wrong output on {case.stdin}"
            return None

        report = run_task(
            {"prompt": "Solve example.", "tests": [public], "held_out": [private]},
            generate, run_test,
        )
        self.assertEqual(len(requests), 2)
        self.assertIn("Wrong output on public input", requests[1][-1]["content"])
        self.assertNotIn(private["stdin"], json.dumps(requests))
        self.assertNotIn(private["stdout"], json.dumps(requests))
        self.assertFalse(report["single_pass_held_out_passed"])
        self.assertTrue(report["checked_held_out_passed"])

    def test_never_executes_a_truncated_reply_and_stops_after_two_attempts(self):
        runner = Mock(return_value="Wrong answer")
        generate = Mock(side_effect=[(DRAFT, "length"), (DRAFT, "stop")])
        results = solve("Example", [Case("", "")], generate, runner)
        self.assertEqual(generate.call_count, 2)
        runner.assert_called_once()
        self.assertIn("complete fenced C program", results[0].failure)
        self.assertEqual(results[-1].failure, "Wrong answer")

    def test_success_avoids_a_second_generation(self):
        runner = Mock(return_value=None)
        generate = Mock(return_value=(FIX, "stop"))
        self.assertIsNone(solve("Example", [Case("", "")], generate, runner)[0].failure)
        generate.assert_called_once()

    @patch("src.checked_c.run_container")
    def test_docker_invocation_uses_no_mounts_and_keeps_answers_on_host(self, run):
        run.return_value = (0, b"actual\n")
        result = sandbox_test("code", Case("my input\n", "my expected\n"))
        self.assertIn("expected 'my expected", result)
        self.assertIn("got 'actual", result)
        command = run.call_args.args[0]
        for option in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never"):
            self.assertIn(option, command)
        self.assertNotIn("--mount", command)
        self.assertNotIn("--volume", command)
        self.assertNotIn("--privileged", command)
        payload = run.call_args.args[1]
        self.assertEqual([base64.b64decode(part) for part in payload.split(b"\n")], [b"code", b"my input\n"])

    @patch("src.checked_c.subprocess.run")
    def test_output_overflow_removes_the_named_container(self, cleanup):
        _, output = run_container(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"],
            b"", "overflow",
        )
        self.assertEqual(len(output), 8193)
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.args[0], ["docker", "rm", "-f", "overflow"])

    def test_uses_local_llama_chat_endpoint(self):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                result = {"choices": [{"message": {"content": FIX}, "finish_reason": "stop"}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.handle_request, daemon=True)
            thread.start()
            messages = [{"role": "user", "content": "Test"}]
            self.assertEqual(request_completion(server.server_port, messages, 64), (FIX, "stop"))
            thread.join(timeout=2)
        self.assertEqual(observed[0][0], "/v1/chat/completions")
        self.assertEqual(observed[0][1]["messages"], messages)
