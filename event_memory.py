"""Store the role's event memory and append its own published expression."""
from copy import deepcopy


def update_event_memory(old, text):
    result = deepcopy(old)
    if text is not None:
        result.text = text
        result.version += 1
    return result


def record_action(memory, step, post):
    result = deepcopy(memory)
    result.actions.append(dict(step=step, post_id=post.post_id, content=post.content))
    result.text = (result.text + f'\nPosted at step {step}: {post.content}').strip()
    result.version += 1
    return result
