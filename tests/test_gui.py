import json
from http.client import HTTPConnection
from threading import Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from src.generate import Completion
from src.gui import ChatServer, ChatSession


class ChatGuiTests(TestCase):
    def setUp(self):
        self.generator = Mock()
        self.generator.tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
        self.generator.tokenizer.tok = None
        self.generator.model.cfg = SimpleNamespace(block_size=144)
        self.generator.device = "cpu"
        self.generator.generate.return_value = [Completion("An answer.", 8, "eos")]
        self.server = ChatServer(0, ChatSession(self.generator, "runs/reasoning/sft/export", 32))
        self.thread = Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, method, path, payload=None, **headers):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            body = None if payload is None else json.dumps(payload)
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    def send(self, turns, origin=None):
        return self.request("POST", "/api/chat", {"messages": turns}, **{
            "Content-Type": "application/json", "Origin": origin or self.origin,
        })

    def test_serves_assets_and_retains_recent_conversation(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn('id="prompt"', page)
        status, model = self.request("GET", "/api/model")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(model)["device"], "cpu")

        status, reply = self.send([
            {"role": "user", "content": "x" * 84},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "Follow-up question"},
        ])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(reply), {"text": "An answer.", "finish_reason": "eos", "omitted_turns": 1})
        prompt = self.generator.generate.call_args.args[0][0]
        self.assertIn("Follow-up question", prompt)
        self.assertNotIn("Old answer", prompt)

    def test_rejects_cross_origin_and_invalid_message_order(self):
        turns = [{"role": "user", "content": "A question"}]
        self.assertEqual(self.send(turns, "http://example.test")[0], 403)
        self.assertEqual(self.request("GET", "/api/model", Host="example.test")[0], 403)
        self.assertEqual(self.send([{"role": "system", "content": "Test"}])[0], 400)
        self.assertEqual(self.send([{"role": "user", "content": "x" * 144}])[0], 400)
        self.generator.generate.assert_not_called()

    def test_keeps_both_sides_of_a_followup_that_fits(self):
        status, reply = self.send([
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Why?"},
        ])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(reply)["omitted_turns"], 0)
        prompt = self.generator.generate.call_args.args[0][0]
        self.assertIn("Question one", prompt)
        self.assertIn("Answer one", prompt)
        self.assertIn("Why?", prompt)
