import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from typing import List

logger = init_logger(__name__)


# get reward
def request_reward(text: List[str], host: str, rm: str, retry_delay=0.2, max_retries=8) -> float:
    if isinstance(text, str):
        text = [text]
    for i in range(max_retries):
        try:
            res = requests.post(
                f"http://{host}/classify",
                json={
                    "model": rm,
                    "text": text,
                },
            )
            rewards = [e['embedding'][0] for e in res.json()]
            return rewards
        except Exception as e:
            print(f"Error requesting reward: {e}")
            time.sleep(retry_delay)
            continue
    print(f"Failed to request reward after {max_retries} retries")
    with open("/cpfs01/shared/llm_ddd/zouyicheng/OpenRLHF/logs/error/error.log", "a", encoding="utf8") as f:
        f.write(f"Text: {text}\n")
    return None


def remote_rm_fn(api_url, queries, score_key="rewards", mean=0.0, std=10.0):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """

    scores = request_reward(queries, api_url, rm=score_key)
    scores = torch.tensor(scores)
    scores = (scores - mean) / std
    return scores


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)
