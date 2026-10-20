from __future__ import annotations

import json
import uuid

from odoo.addons.muk_ai.providers.base import ProviderBase
from odoo.addons.muk_ai.providers.region import CUSTOM


class OllamaFastAPIProvider(ProviderBase):
    name = 'ollama_fastapi'
    label = 'Ollama (FastAPI)'
    # An empty value lets the gateway apply OLLAMA_DEFAULT_MODEL until the
    # administrator synchronizes and selects a catalogued model.
    default_model = ''
    default_url = 'http://fastapi:8000/v1'
    regions = (CUSTOM,)
    modalities = frozenset({'chat'})
    supports_vision = False
    supports_web_search = False
    supports_code_interpreter = False

    @property
    def authenticated(self):
        return True

    def headers(self):
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'
        return headers

    def request(self, inputs, tools_schema=None, text_schema=None, on_delta=None,
                model=None, enable_web_search=False,
                enable_code_interpreter=False, extra=None):
        del text_schema, enable_web_search, enable_code_interpreter, extra
        model = self.model_for(model)
        body = {
            'model': model,
            'messages': self._messages(inputs),
            'max_tokens': self.max_tokens,
        }
        if tools_schema:
            body['tools'] = [self._tool_schema(tool) for tool in tools_schema]
        if callable(on_delta) and self.provider.ollama_streaming:
            return self._stream_chat(body, on_delta, model)
        return self._parse(self._post_json('/chat/completions', body))

    @staticmethod
    def _text(content):
        if isinstance(content, str):
            return content
        return '\n'.join(
            block.get('text', '') for block in (content or [])
            if isinstance(block, dict) and block.get('type') in ('input_text', 'output_text', 'text')
        )

    @classmethod
    def _messages(cls, inputs):
        messages = []
        pending_calls = {}
        for item in cls._wire_items(inputs):
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            if item_type == 'function_call':
                call_id = item.get('call_id') or item.get('id') or f'call_{uuid.uuid4().hex}'
                call = {
                    'id': call_id,
                    'type': 'function',
                    'function': {
                        'name': item.get('name') or '',
                        'arguments': item.get('arguments') or '{}',
                    },
                }
                if messages and messages[-1]['role'] == 'assistant':
                    messages[-1].setdefault('tool_calls', []).append(call)
                else:
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
                pending_calls[call_id] = call
            elif item_type == 'function_call_output':
                messages.append({
                    'role': 'tool',
                    'tool_call_id': item.get('call_id'),
                    'content': item.get('output') or '',
                })
            elif role := item.get('role'):
                message = {'role': role, 'content': cls._text(item.get('content'))}
                if item.get('tool_calls'):
                    message['tool_calls'] = item['tool_calls']
                messages.append(message)
        return messages

    @staticmethod
    def _tool_schema(schema):
        if schema.get('function'):
            return schema
        return {
            'type': 'function',
            'function': {
                key: value for key, value in schema.items()
                if key in ('name', 'description', 'parameters')
            },
        }

    def _parse(self, payload):
        choice = next(iter(payload.get('choices') or []), {})
        message = choice.get('message') or {}
        text = message.get('content') or ''
        calls = self._calls(message.get('tool_calls'))
        carry = self._carry(text, calls)
        usage = payload.get('usage') or {}
        return {
            'text': text.strip(),
            'tool_calls': calls,
            'carry_inputs': carry,
            'usage': self._usage(
                input_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'),
            ),
        }

    def _stream_chat(self, body, on_delta, model):
        body['stream'] = True
        text_parts, calls_by_index, usage = [], {}, {}
        for event in self._post_stream('/chat/completions', body):
            if error := event.get('error'):
                self._raise(error.get('message') or str(error))
            usage = event.get('usage') or usage
            choice = next(iter(event.get('choices') or []), {})
            delta = choice.get('delta') or {}
            if text := delta.get('content'):
                text_parts.append(text)
                self._call_on_delta(on_delta, 'text', {'delta': text})
            for call in delta.get('tool_calls') or []:
                index = call.get('index', 0)
                entry = calls_by_index.setdefault(index, {
                    'id': call.get('id'), 'name': '', 'arguments': ''
                })
                entry['id'] = call.get('id') or entry['id']
                function = call.get('function') or {}
                if function.get('name'):
                    entry['name'] = function['name']
                    self._call_on_delta(on_delta, 'tool_start', {
                        'call_id': entry['id'], 'name': entry['name']
                    })
                if arguments := function.get('arguments'):
                    entry['arguments'] += arguments
                    self._call_on_delta(on_delta, 'tool_args', {
                        'call_id': entry['id'], 'delta': arguments
                    })
        text = ''.join(text_parts)
        raw_calls = [{
            'id': value['id'], 'type': 'function',
            'function': {'name': value['name'], 'arguments': value['arguments']}
        } for value in calls_by_index.values()]
        calls = self._calls(raw_calls)
        return {
            'text': text.strip(),
            'tool_calls': calls,
            'carry_inputs': self._carry(text, calls),
            'usage': self._usage(usage.get('prompt_tokens'), usage.get('completion_tokens')),
        }

    def _calls(self, raw_calls):
        result = []
        for raw in raw_calls or []:
            function = raw.get('function') or {}
            arguments, parse_error = self._parse_tool_arguments(function.get('arguments'))
            result.append({
                'call_id': raw.get('id') or f'call_{uuid.uuid4().hex}',
                'name': function.get('name'),
                'arguments': arguments,
                '_parse_error': parse_error,
            })
        return result

    @staticmethod
    def _carry(text, calls):
        carry = []
        if text:
            carry.append({'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]})
        carry.extend({
            'type': 'function_call',
            'call_id': call['call_id'],
            'name': call['name'],
            'arguments': json.dumps(call['arguments'], separators=(',', ':')),
        } for call in calls)
        return carry
