import json
from unittest.mock import Mock, patch

from odoo.tests import tagged
from odoo.tests.common import TransactionCase


@tagged('post_install', '-at_install')
class TestOllamaProvider(TransactionCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.provider = cls.env.ref('muk_ai_ollama_fastapi.provider_ollama_fastapi')

    def test_connection_and_chat_contract(self):
        client = self.provider._get_client()
        payload = {'choices': [{'message': {'content': 'hello'}}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}
        with patch.object(client, '_post_json', return_value=payload):
            result = client.request([{'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}])
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(result['carry_inputs'][0]['role'], 'assistant')
        self.assertEqual(result['usage']['input_tokens'], 2)

    def test_streaming_and_tool_call(self):
        client = self.provider._get_client()
        events = [
            {'choices': [{'delta': {'content': 'Hi '}}]},
            {'choices': [{'delta': {'content': 'there'}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'search', 'arguments': '{"q":"x"}'}}]}}]},
            {'choices': [{'delta': {}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}},
        ]
        deltas = []
        with patch.object(client, '_post_stream', return_value=iter(events)):
            result = client.request([], tools_schema=[{'type': 'function', 'name': 'search', 'parameters': {}}], on_delta=lambda kind, data: deltas.append((kind, data)))
        self.assertEqual(result['text'], 'Hi there')
        self.assertEqual(result['tool_calls'][0]['arguments'], {'q': 'x'})
        self.assertTrue(any(kind == 'text' for kind, _data in deltas))

    def test_model_synchronization(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {'data': [{'id': 'llama-test:latest'}]}
        client = self.provider._get_client()
        with patch.object(type(client), '_http_session', return_value=Mock(get=Mock(return_value=response))):
            self.provider.action_sync_ollama_models()
        model = self.env['muk_ai.model'].search([('provider_id', '=', self.provider.id), ('technical_name', '=', 'llama-test:latest')])
        self.assertTrue(model)

    def test_existing_session_path_uses_provider(self):
        self.assertEqual(self.provider._get_client().name, 'ollama_fastapi')
        self.assertTrue(self.provider.serves_chat)
