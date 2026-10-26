"""Create an OpenAI-compatible client from environment credentials."""
import os


def create_provider_client(model, **options):
    from openai import AsyncOpenAI
    if model.startswith('deepseek'):
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        base_url = 'https://api.deepseek.com/v1'
    elif model.startswith('qwen'):
        api_key = os.environ.get('QWEN_API_KEY', '')
        base_url = os.environ.get('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    else:
        api_key = os.environ.get('OPENAI_API_KEY', '')
        base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    if not api_key.strip():
        required = ('DEEPSEEK_API_KEY' if model.startswith('deepseek') else
                    'QWEN_API_KEY' if model.startswith('qwen') else 'OPENAI_API_KEY')
        raise ValueError(f'Missing API credentials: set {required} before execution')
    return AsyncOpenAI(api_key=api_key, base_url=base_url, **options)
