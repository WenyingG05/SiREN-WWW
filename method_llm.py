"""Three role calls; API retries and empty-result fallback live here."""
import asyncio
import json
import time
from pathlib import Path
from method_parsing import parse_json, parse_cognition, bounded_float
from method_prompts import load_prompts


class APIRefusal(RuntimeError):
    """The provider blocked the current request or response."""


class APITransport:
    def __init__(self, model, temperature, max_tokens, timeout_seconds):
        from llm_transport import create_provider_client
        self.client = create_provider_client(model, max_retries=0, timeout=timeout_seconds)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_metadata = {}

    async def call(self, system, user, json_mode=True):
        options = {}
        if json_mode:
            options['response_format'] = {'type': 'json_object'}
        if self.model.startswith('qwen'):
            options['extra_body'] = {'enable_thinking': False}
        response = await self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], **options)
        choice = response.choices[0]
        self.last_metadata = dict(response_id=response.id, response_model=response.model,
                                  usage=response.usage.model_dump() if response.usage else None)
        if choice.finish_reason == 'content_filter' or getattr(choice.message, 'refusal', None):
            raise APIRefusal('provider content filter or refusal')
        return choice.message.content or ''


def retryable(exc):
    if isinstance(exc, APIRefusal):
        return False
    status = getattr(exc, 'status_code', None)
    if status is not None:
        return status in {408, 409, 429} or 500 <= status < 600
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        'APITimeoutError', 'APIConnectionError'}


class MethodLLM:
    def __init__(self, transport, language='zh', max_attempts=3, retry_delay_seconds=1., prompt_dir=None):
        self.transport = transport
        self.language = language
        root = Path(prompt_dir) if prompt_dir else Path(__file__).parent / 'prompts'
        self.prompts = load_prompts(root / f'method_{language}.json')
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay_seconds
        self.io_log = []

    async def request(self, stage, payload, parser=None):
        system = self.prompts[stage]
        user = json.dumps(payload, ensure_ascii=False)
        raw = ''
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            entry = dict(stage=stage, attempt=attempt, system=system, user=user)
            self.io_log.append(entry)
            try:
                raw = await self.transport.call(system, user, json_mode=stage != 'generation')
                entry.update(response=raw, status='ok',
                             provider=dict(getattr(self.transport, 'last_metadata', {})))
                break
            except Exception as exc:
                retry = retryable(exc) and attempt < self.max_attempts
                entry.update(status='api_error', error_type=type(exc).__name__, error=str(exc), retry=retry)
                if not retry:
                    return None
                await asyncio.sleep(self.retry_delay * 2 ** (attempt - 1))
            finally:
                entry['elapsed_seconds'] = time.monotonic() - started
        try:
            if not raw.strip():
                entry.update(status='empty_response', error='empty model response')
                return None
            return parser(parse_json(raw)) if parser else raw.strip()
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            entry.update(status='parse_error', error=str(exc))
            return None

    async def cognize(self, posts, state, max_cues):
        payload = dict(posts=[dict(author=p.author_id, content=p.content,
                                   time=p.real_time or p.step) for p in posts],
                       profile=state.profile, graph=state.graph.context(),
                       event_memory=state.memory.text, mood=state.mood, max_cues_per_batch=max_cues)
        return await self.request('cognition', payload, lambda d: parse_cognition(d, max_cues))

    async def posting_probability(self, payload):
        return await self.request('action', payload, lambda d: bounded_float(d['posting_probability']))

    async def generate(self, payload):
        return await self.request('generation', payload)
