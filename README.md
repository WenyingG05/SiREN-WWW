# SiREN

Anonymous code release for a social media risk belief and emotion simulation.

## Setup

Use Python 3.10 or newer. Install the dependencies (choose a PyTorch build appropriate for your machine if needed):

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env`. If your chat model is served by an OpenAI-compatible provider, set `OPENAI_BASE_URL` as well. Edit `llm.model` in `configs/example.yaml` to the model available at that endpoint. The example embedding model is downloaded from Hugging Face on first use; `cognition.embedding_model` may instead point to a local compatible model directory.

## Run the synthetic example

```bash
python run.py --config configs/example.yaml
```

The example has two synthetic English-language users, one synthetic media notice, and two simulation steps. Step 0 introduces the notice; agents read it at step 1. Run files are written under `output/example/`, which is excluded from version control. A live run requires a reachable chat model and embedding model.

## Use your own inputs

Start from `configs/example.yaml`. Each entry under `agents` needs an `agent_id`, `center_risk_object`, and optional `profile`, `event_memory`, `belief_graph`, and `mood`. Set `language` to `en` or `zh`. Each entry under `media.records` needs `record_id`, `media_agent_id`, `content`, and `published_step`. A post published at step *t* is read by followers at step *t + 1*. For larger runs, `initialization_package_path` can point to a JSON object containing an `agents` list; `network.precomputed_path` can point to a JSON object containing `node_to_agent`, `peer_edges_undirected`, and `media_to_personal_arcs`.

The simulation reads each user's followed posts in one daily batch. The LLM returns risk relevance, cues, appraised beliefs, local belief graph changes, and event memory. The program retrieves old beliefs, computes Fear from current and activated beliefs, updates Mood, then asks the LLM for a posting probability and, if sampled, a post. State and step logs are saved so a run can be resumed with `resume_checkpoint`.

## Prompts and outputs

`prompts/method_en.json` and `prompts/method_zh.json` are the general prompts used at runtime. `method_prompts.py` joins their shared context, stage definitions, and stage task into the system message for cognition, action, or generation.

Each run directory contains a configuration snapshot, method manifest with source and prompt hashes, initial states, network, step logs, checkpoints, generated posts, and a summary. The logs include model inputs and outputs; review them before sharing a run directory.
