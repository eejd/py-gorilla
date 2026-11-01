"""Regression coverage for the Cohere 7 and Mistral 3 BFCL adapters."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import cohere
from mistralai.client import Mistral
from bfcl_eval.model_handler.api_inference.cohere import CohereHandler
from bfcl_eval.model_handler.api_inference.mistral import MistralHandler


@contextmanager
def response_server(response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_mistral_tool_response(monkeypatch):
    monkeypatch.setenv('MISTRAL_API_KEY', 'local')
    response = {'id': 'chat', 'object': 'chat.completion', 'created': 0, 'model': 'local',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None,
                    'tool_calls': [{'id': 'call-1', 'type': 'function', 'function':
                        {'name': 'add', 'arguments': '{"a":19,"b":23}'}}]},
                    'finish_reason': 'tool_calls'}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}
    with response_server(response) as (url, requests):
        handler = MistralHandler('local', 0, 'local-FC', True)
        handler.client = Mistral(api_key='local', server_url=url)
        result, _ = handler.generate_with_backoff(model='local', messages=[{'role': 'user', 'content': 'hello'}], temperature=0)
        parsed = handler._parse_query_response_FC(result)
        assert handler.decode_ast(parsed['model_responses'], 'Python', False) == [{'add': {'a': 19, 'b': 23}}]
        assert requests[0]['messages'][0]['content'] == 'hello'


def test_cohere_content_response(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'local')
    response = {'id': 'test', 'finish_reason': 'COMPLETE', 'message': {'role': 'assistant',
                'content': [{'type': 'thinking', 'thinking': 'work'}, {'type': 'text', 'text': '42'}]},
                'usage': {'billed_units': {'input_tokens': 2, 'output_tokens': 1}}}
    with response_server(response) as (url, requests):
        handler = CohereHandler('local', 0, 'local-FC', True)
        handler.client = cohere.ClientV2(api_key='local', base_url=url)
        data = {'chat_turns': [cohere.UserChatMessageV2(role='user', content='hello')], 'tools': []}
        result, _ = handler._query_FC(data)
        parsed = handler._parse_query_response_FC(result)
        assert parsed['model_responses'] == '42'
        assert parsed['reasoning_content'] == 'work'
        assert requests[0]['messages'][0]['content'] == 'hello'
