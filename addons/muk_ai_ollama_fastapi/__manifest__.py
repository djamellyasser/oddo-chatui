{
    'name': 'MuK AI Ollama FastAPI',
    'summary': 'Local Ollama models through an OpenAI-compatible FastAPI gateway',
    'version': '19.0.1.0.0',
    'category': 'Productivity',
    'license': 'LGPL-3',
    'depends': ['muk_ai'],
    'data': [
        'security/security.xml',
        'data/provider.xml',
        'views/provider.xml',
    ],
    'installable': True,
    'application': False,
}
