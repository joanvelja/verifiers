import asyncio
from collections import OrderedDict
from typing import Any

from verifiers.clients import Client
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_evidence_cache import (
    JudgeDecision,
    JudgeEvidenceSample,
    JudgePanelPolicy,
    SQLiteJudgeEvidenceCache,
    canonical_text,
    stable_hash,
    verdict_token,
)
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ClientConfig, Messages, State, SystemMessage, UserMessage

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    def __init__(
        self,
        parser: Parser | None = None,
        judge_client: Client | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_system_prompt: str | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        judge_cache_enabled: bool = True,
        judge_cache_size: int = 10000,
        judge_max_retries: int = 2,
        judge_retry_delay_s: float = 0.5,
        judge_persistent_cache_path: str | None = None,
        judge_persistent_cache_min_samples: int = 1,
        judge_rubric_family: str = "default",
        judge_variant_id: str = "default",
        judge_positive_label: str = "YES",
        judge_negative_label: str = "NO",
        judge_reward_mode: str = "hard",
        judge_panel_threshold: float = 0.5,
        judge_panel_prior_alpha: float = 1.0,
        judge_panel_prior_beta: float = 1.0,
        judge_repeated_call_correlation: float = 0.0,
        judge_calibration_mode: str = "vote_fraction",
        judge_correctness_prior: float = 0.5,
        judge_sensitivity: float | None = None,
        judge_false_positive_rate: float | None = None,
    ):
        super().__init__(parser=parser)
        self.judge_client: Client = (
            judge_client
            if judge_client is not None
            else OpenAIChatCompletionsClient(ClientConfig())
        )
        self.judge_model = judge_model
        self.judge_system_prompt = judge_system_prompt
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}
        self.judge_cache_enabled = judge_cache_enabled
        self.judge_cache_size = max(0, judge_cache_size)
        self.judge_max_retries = max(0, judge_max_retries)
        self.judge_retry_delay_s = max(0.0, judge_retry_delay_s)
        self.judge_persistent_cache_path = judge_persistent_cache_path
        self.judge_persistent_cache_min_samples = max(
            1, judge_persistent_cache_min_samples
        )
        self.judge_rubric_family = judge_rubric_family
        self.judge_variant_id = judge_variant_id
        self.judge_positive_label = judge_positive_label
        self.judge_negative_label = judge_negative_label
        self.judge_reward_mode = judge_reward_mode
        self.judge_calibration_mode = judge_calibration_mode
        self.judge_correctness_prior = judge_correctness_prior
        self.judge_sensitivity = judge_sensitivity
        self.judge_false_positive_rate = judge_false_positive_rate
        self.judge_panel = JudgePanelPolicy(
            positive_label=judge_positive_label,
            negative_label=judge_negative_label,
            prior_alpha=judge_panel_prior_alpha,
            prior_beta=judge_panel_prior_beta,
            threshold=judge_panel_threshold,
            reward_mode=judge_reward_mode,
            repeated_call_correlation=judge_repeated_call_correlation,
            calibration_mode=judge_calibration_mode,
            correctness_prior=judge_correctness_prior,
            judge_sensitivity=judge_sensitivity,
            judge_false_positive_rate=judge_false_positive_rate,
        )
        self._persistent_cache = (
            SQLiteJudgeEvidenceCache(judge_persistent_cache_path)
            if judge_persistent_cache_path is not None
            else None
        )
        self.judge_cache_stats = {
            "hits": 0,
            "misses": 0,
            "inflight_hits": 0,
            "evictions": 0,
            "retries": 0,
            "persistent_hits": 0,
            "persistent_misses": 0,
            "persistent_writes": 0,
            "persistent_decision_hits": 0,
        }
        self._judge_cache: OrderedDict[str, str] = OrderedDict()
        self._judge_inflight: dict[str, asyncio.Task[str]] = {}
        self._judge_cache_lock = asyncio.Lock()
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
            "judge_system_prompt": self.judge_system_prompt,
            "judge_prompt": self.judge_prompt,
            "judge_sampling_args": self.judge_sampling_args,
            "judge_cache_enabled": self.judge_cache_enabled,
            "judge_cache_size": self.judge_cache_size,
            "judge_max_retries": self.judge_max_retries,
            "judge_retry_delay_s": self.judge_retry_delay_s,
            "judge_persistent_cache_path": self.judge_persistent_cache_path,
            "judge_persistent_cache_min_samples": self.judge_persistent_cache_min_samples,
            "judge_rubric_family": self.judge_rubric_family,
            "judge_variant_id": self.judge_variant_id,
            "judge_positive_label": self.judge_positive_label,
            "judge_negative_label": self.judge_negative_label,
            "judge_reward_mode": self.judge_reward_mode,
            "judge_calibration_mode": self.judge_calibration_mode,
            "judge_correctness_prior": self.judge_correctness_prior,
            "judge_sensitivity": self.judge_sensitivity,
            "judge_false_positive_rate": self.judge_false_positive_rate,
            "judge_panel_threshold": judge_panel_threshold,
            "judge_panel_prior_alpha": judge_panel_prior_alpha,
            "judge_panel_prior_beta": judge_panel_prior_beta,
            "judge_repeated_call_correlation": judge_repeated_call_correlation,
        }

    def _normalize_judge_args(self) -> dict[str, Any]:
        judge_args = dict(self.judge_sampling_args or {})
        if "max_tokens" in judge_args:
            if judge_args["max_tokens"] is None:
                judge_args.pop("max_tokens")
            else:
                judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
        if (
            "max_completion_tokens" in judge_args
            and judge_args["max_completion_tokens"] is None
        ):
            judge_args.pop("max_completion_tokens")
        return {k: v for k, v in judge_args.items() if v is not None}

    def _make_state_cache_key(
        self, judge_system_prompt: str | None, judge_prompt: str
    ) -> str:
        return (
            f"system:\n{judge_system_prompt}\n\nuser:\n{judge_prompt}"
            if judge_system_prompt
            else judge_prompt
        )

    def _make_global_cache_key(
        self,
        *,
        judge_system_prompt: str | None,
        judge_prompt: str,
        judge_args: dict[str, Any],
    ) -> str:
        payload = {
            "model": self.judge_model,
            "sampling_args": judge_args,
            "system": judge_system_prompt,
            "user": judge_prompt,
        }
        return stable_hash(payload)

    def _make_persistent_identity(
        self,
        *,
        question: str,
        answer: str,
        response: str,
        state: State | None,
    ) -> tuple[str, dict[str, str | None]]:
        question_id = None
        if state is not None and state.get("example_id") is not None:
            question_id = canonical_text(state["example_id"])
        canonical_question = canonical_text(question)
        identity = {
            "rubric_family": canonical_text(self.judge_rubric_family),
            "question_id": question_id,
            "question_hash": stable_hash({"question": canonical_question}),
            "target": canonical_text(answer),
            "response": canonical_text(response),
        }
        key_payload = {
            "rubric_family": identity["rubric_family"],
            "question_id": question_id,
            "question_hash": identity["question_hash"] if question_id is None else None,
            "target": identity["target"],
            "response": identity["response"],
        }
        return stable_hash(key_payload), identity

    def _save_state_cache(
        self,
        state: State | None,
        *,
        cache_key: str,
        judge_response: str,
    ) -> None:
        if state is None:
            return
        cached = state.get("judge_response")
        if not isinstance(cached, dict):
            cached = {}
        cached[cache_key] = judge_response
        state["judge_response"] = cached

    def _save_state_decision(
        self,
        state: State | None,
        *,
        cache_key: str,
        decision: JudgeDecision,
    ) -> None:
        if state is None:
            return
        cached = state.get("judge_decision")
        if not isinstance(cached, dict):
            cached = {}
        decision_dict = decision.as_dict()
        cached[cache_key] = decision_dict
        state["judge_decision"] = cached
        state["judge_decision_last"] = decision_dict

    def _decision_from_raw_response(self, raw_response: str) -> JudgeDecision:
        return self.judge_panel.decide(
            [
                JudgeEvidenceSample(
                    raw_response=raw_response,
                    verdict=verdict_token(raw_response),
                )
            ]
        )

    async def _store_global_cache(self, cache_key: str, judge_response: str) -> None:
        if not self.judge_cache_enabled or self.judge_cache_size == 0:
            return
        async with self._judge_cache_lock:
            self._judge_cache[cache_key] = judge_response
            self._judge_cache.move_to_end(cache_key)
            while len(self._judge_cache) > self.judge_cache_size:
                self._judge_cache.popitem(last=False)
                self.judge_cache_stats["evictions"] += 1

    async def _call_judge_backend(
        self,
        *,
        judge_messages: Messages,
        judge_args: dict[str, Any],
        state: State | None,
    ) -> str:
        last_error: BaseException | None = None
        for attempt in range(self.judge_max_retries + 1):
            try:
                response_obj = await self.judge_client.get_response(
                    prompt=judge_messages,
                    model=self.judge_model,
                    sampling_args=judge_args,
                    state=state,
                )
                return str(response_obj.message.content or "")
            except Exception as exc:
                last_error = exc
                if attempt >= self.judge_max_retries:
                    break
                self.judge_cache_stats["retries"] += 1
                await asyncio.sleep(self.judge_retry_delay_s * (2**attempt))
        assert last_error is not None
        raise last_error

    async def _get_or_call_judge(
        self,
        *,
        cache_key: str,
        judge_messages: Messages,
        judge_args: dict[str, Any],
        state: State | None,
        allow_memory_cache: bool = True,
        dedupe_inflight: bool = True,
    ) -> str:
        if not self.judge_cache_enabled or self.judge_cache_size == 0:
            return await self._call_judge_backend(
                judge_messages=judge_messages,
                judge_args=judge_args,
                state=state,
            )
        if not dedupe_inflight:
            return await self._call_judge_backend(
                judge_messages=judge_messages,
                judge_args=judge_args,
                state=state,
            )

        owns_task = False
        async with self._judge_cache_lock:
            if allow_memory_cache:
                cached = self._judge_cache.get(cache_key)
                if cached is not None:
                    self._judge_cache.move_to_end(cache_key)
                    self.judge_cache_stats["hits"] += 1
                    return cached

            task = self._judge_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._call_judge_backend(
                        judge_messages=judge_messages,
                        judge_args=judge_args,
                        state=state,
                    )
                )
                self._judge_inflight[cache_key] = task
                self.judge_cache_stats["misses"] += 1
                owns_task = True
            else:
                self.judge_cache_stats["inflight_hits"] += 1

        try:
            judge_response = await task
        finally:
            if owns_task:
                async with self._judge_cache_lock:
                    self._judge_inflight.pop(cache_key, None)

        if owns_task and allow_memory_cache:
            await self._store_global_cache(cache_key, judge_response)
        return judge_response

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        if isinstance(prompt, list):
            last_msg = prompt[-1]
            if isinstance(last_msg, dict):
                question = str(last_msg.get("content", ""))
            else:
                question = str(getattr(last_msg, "content", ""))
        else:
            question = str(prompt)
        response = self.parser.parse_answer(completion)
        render_vars = {"question": question, "answer": answer, "response": response}
        judge_prompt = self.judge_prompt.format(**render_vars)
        judge_system_prompt = (
            self.judge_system_prompt.format(**render_vars)
            if self.judge_system_prompt
            else None
        )
        state_cache_key = self._make_state_cache_key(judge_system_prompt, judge_prompt)
        cached = state.get("judge_response") if state else None
        if isinstance(cached, dict) and state_cache_key in cached:
            decisions = state.get("judge_decision") if state else None
            if isinstance(decisions, dict) and state_cache_key in decisions:
                decision = decisions[state_cache_key]
                if isinstance(decision, dict) and decision.get("hard_response"):
                    return str(decision["hard_response"])
            return cached[state_cache_key]

        judge_args = self._normalize_judge_args()

        judge_messages: Messages = []
        if judge_system_prompt:
            judge_messages.append(SystemMessage(content=judge_system_prompt))
        judge_messages.append(UserMessage(content=judge_prompt))
        persistent_cache_key, persistent_identity = self._make_persistent_identity(
            question=question,
            answer=answer,
            response=response or "",
            state=state,
        )
        persistent_needs_fresh_sample = False
        if self._persistent_cache is not None:
            samples = self._persistent_cache.get_samples(
                persistent_cache_key, self.judge_variant_id
            )
            if len(samples) >= self.judge_persistent_cache_min_samples:
                decision = self.judge_panel.decide(samples)
                self.judge_cache_stats["persistent_hits"] += 1
                self.judge_cache_stats["persistent_decision_hits"] += 1
                await self._store_global_cache(
                    self._make_global_cache_key(
                        judge_system_prompt=judge_system_prompt,
                        judge_prompt=judge_prompt,
                        judge_args=judge_args,
                    ),
                    decision.raw_response,
                )
                self._save_state_cache(
                    state,
                    cache_key=state_cache_key,
                    judge_response=decision.raw_response,
                )
                self._save_state_decision(
                    state,
                    cache_key=state_cache_key,
                    decision=decision,
                )
                return decision.hard_response
            else:
                persistent_needs_fresh_sample = True
            self.judge_cache_stats["persistent_misses"] += 1

        global_cache_key = self._make_global_cache_key(
            judge_system_prompt=judge_system_prompt,
            judge_prompt=judge_prompt,
            judge_args=judge_args,
        )
        judge_response = await self._get_or_call_judge(
            cache_key=global_cache_key,
            judge_messages=judge_messages,
            judge_args=judge_args,
            state=state,
            allow_memory_cache=not persistent_needs_fresh_sample,
            dedupe_inflight=not persistent_needs_fresh_sample,
        )
        decision = self._decision_from_raw_response(judge_response)
        if self._persistent_cache is not None:
            self._persistent_cache.add_sample(
                cell_key=persistent_cache_key,
                identity=persistent_identity,
                variant_id=self.judge_variant_id,
                raw_response=judge_response,
                grader_model=self.judge_model,
                sampling_args=judge_args,
                judge_prompt=judge_prompt,
                judge_system_prompt=judge_system_prompt,
            )
            self.judge_cache_stats["persistent_writes"] += 1

        self._save_state_cache(
            state,
            cache_key=state_cache_key,
            judge_response=judge_response,
        )
        self._save_state_decision(
            state,
            cache_key=state_cache_key,
            decision=decision,
        )
        return decision.hard_response
