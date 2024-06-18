# refer 
# - https://github.com/shwu-nyunai/lm-evaluation-harness/blob/main/lm_eval/models/nemo_lm.py
# - https://github.com/shwu-nyunai/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
# - https://github.com/shwu-nyunai/lm-evaluation-harness/blob/main/lm_eval/models/gguf.py

import logging

from tqdm import tqdm
from typing import List, Tuple

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

# qserve
from qserve import EngineArgs, LLMEngine, SamplingParams

logger = logging.getLogger(__name__)


def initialize_engine(engine_args: EngineArgs) -> LLMEngine:
    """Initialize the LLMEngine from the model-args."""
    return LLMEngine.from_engine_args(engine_args)


@register_model("mit_han_qserve", "qserve", "mit-han-qserve")
class MIT_HAN_QServe(LM):
    def __init__(
        self,
        **kwargs,  # catch everything representing - https://github.com/mit-han-lab/qserve/tree/main?tab=readme-ov-file#usage-and-examples
    ):

        super().__init__()
        engine_args = EngineArgs(**kwargs)
        self.engine = initialize_engine(engine_args)

    def tok_encode(
        self, string: str, **kwargs
    ) -> List[int]:
        return self.engine.tokenizer.encode(string, **kwargs)

    def _engine_completion(self, context: str, continuation: str, **kwargs):
        pass

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            response = self._engine_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain 'token_logprobs' list."
                    )
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                assert False
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
