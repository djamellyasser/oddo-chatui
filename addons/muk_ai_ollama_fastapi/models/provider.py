import requests

from odoo import _, fields, models
from odoo.exceptions import UserError


class AIProvider(models.Model):
    _inherit = 'muk_ai.provider'

    ollama_streaming = fields.Boolean(
        string='Enable Streaming', default=True,
        help='Stream model output through the existing MuK AI session bus.',
        groups='base.group_system',
    )

    def action_test_connection(self):
        """Discover the local catalog before MuK chooses a test model."""
        self.ensure_one()
        if self.name == 'ollama_fastapi' and not self.default_chat_model_id:
            self.action_sync_ollama_models()
        return super().action_test_connection()

    def action_sync_ollama_models(self):
        self.ensure_one()
        if self.name != 'ollama_fastapi':
            raise UserError(_('Model synchronization is only available for Ollama.'))
        client = self._get_client()
        try:
            response = client._http_session().get(
                f'{client.api_url}/models', headers=client.headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            entries = response.json().get('data') or []
        except (requests.RequestException, ValueError) as exc:
            raise UserError(_('Unable to synchronize Ollama models: %s', exc)) from exc
        found = self.env['muk_ai.model']
        for entry in entries:
            technical_name = entry.get('id')
            if not technical_name:
                continue
            model = self.env['muk_ai.model'].search([
                ('provider_id', '=', self.id),
                ('technical_name', '=', technical_name),
                ('modality', '=', 'chat'),
            ], limit=1)
            values = {
                'name': technical_name,
                'technical_name': technical_name,
                'provider_id': self.id,
                'modality': 'chat',
                # /v1/models may expose this extension. When Ollama cannot
                # report it, retain an existing administrator value or use a
                # conservative positive placeholder without claiming a size.
                'context_window': entry.get('context_window') or model.context_window or 1,
                'input_rate': 0,
                'output_rate': 0,
                'currency': 'USD',
                'active': True,
            }
            if model:
                model.write(values)
            else:
                model = self.env['muk_ai.model'].create(values)
            found |= model
        if found and not self.default_chat_model_id:
            self.default_chat_model_id = found[0]
        return {
            'type': 'ir.actions.client', 'tag': 'display_notification',
            'params': {'type': 'success', 'title': _('Ollama Models'),
                       'message': _('%s model(s) synchronized.', len(found)),
                       'sticky': False},
        }
