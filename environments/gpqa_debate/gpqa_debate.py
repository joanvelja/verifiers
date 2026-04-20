"""GPQA under a two-debater + judge protocol.

A ``vf-eval``-discoverable environment. Wraps ``DebateEnv`` with:
  * GPQA dataset loading (shuffled A/B/C/D)
  * A default 5-slot propose → critique → final schedule matching
    the ``selfplay`` pack's phase names
  * Per-agent OpenRouter client overrides driven by ``--env-args`` JSON

Launch
------
    export HF_TOKEN=...                  # GPQA is gated
    export OPENROUTER_API_KEY=...

    vf-install gpqa_debate               # once
    vf-eval gpqa_debate                  # minimal run
        --provider openrouter
        --model openai/gpt-4.1-mini
        -n 50 -r 4

    # with per-agent overrides
    vf-eval gpqa_debate
        --provider openrouter
        --model openai/gpt-4.1-mini
        -n 50 -r 4
        --env-args '{
            "prompts_ref": "selfplay",
            "judge_model": "anthropic/claude-3.5-sonnet",
            "debater_providers": ["anthropic", "deepinfra"],
            "judge_providers":   ["anthropic"],
            "allow_fallbacks": false,
            "num_eval_examples": 50,
            "subset": "gpqa_diamond"
        }'

env_args schema
---------------
All optional. Any omitted keys inherit the vf-eval top-level model/provider.

    prompts_ref        str      built-in name or /path/to/pack.yaml   "selfplay"
    subset             str      gpqa_diamond | gpqa_main              "gpqa_diamond"
    num_train_examples int      dataset slice (-1 = all)              -1
    num_eval_examples  int      dataset slice (-1 = all)              -1
    seed               int      choice-shuffling seed                 0
    schedule           list     override SCHEDULE; list of dicts      DEFAULT_SCHEDULE
    truth_member       str      member credited as winning seat        "debater_a"

    # Self-play / OpenRouter overrides (each optional; mutually exclusive
    # with the learner-vs-fixed kwargs below)
    debater_model      str      slug for both debaters                 None (use default)
    judge_model        str      slug for the judge                     None (use default)
    debater_providers  list[str]  strict provider order for debaters   None (OR chooses)
    judge_providers    list[str]  strict provider order for the judge  None
    allow_fallbacks    bool     if true, OR may fall back to non-listed  False

    # Learner-vs-fixed training (set opponent_model to enable)
    opponent_model       str   frozen-opponent model id                None (self-play)
    opponent_base_url    str   OpenAI-compatible endpoint              None (api.openai.com)
    opponent_api_key_var str   env var holding opponent API key        "OPENAI_API_KEY"
    judge_base_url       str   judge endpoint (fallback: opponent)     None (api.openai.com)
    judge_api_key_var    str   env var holding judge API key           "OPENAI_API_KEY"
    learner_seat_mode    str   bernoulli|round_robin|hash|fixed_a|_b   "bernoulli"
    learner_seat_seed    int   RNG seed for bernoulli mode             0
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.clients import Client
from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from verifiers.envs.debate_env import DebateEnv, load_environment as _debate_load_env
from verifiers.types import ClientConfig, State


LETTERS = ("A", "B", "C", "D")

DEFAULT_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "critique"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _format_row(
    row: dict,
    rng: random.Random,
    *,
    example_idx: int,
    learner_seat_mode: str | None,
    seat_rng: random.Random | None,
) -> dict:
    correct = row["Correct Answer"].strip()
    choices = [
        correct,
        row["Incorrect Answer 1"].strip(),
        row["Incorrect Answer 2"].strip(),
        row["Incorrect Answer 3"].strip(),
    ]
    rng.shuffle(choices)
    truth_letter = LETTERS[choices.index(correct)]
    body = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, choices))
    example_id = str(row.get("Record ID") or f"gpqa_{rng.random():.10f}")
    info: dict[str, Any] = {}
    if learner_seat_mode is not None:
        info["learner_seat"] = _assign_learner_seat(
            example_idx=example_idx,
            example_id=example_id,
            mode=learner_seat_mode,
            rng=seat_rng,
        )
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{row['Question'].strip()}\n\n{body}\n\n"
                    "Reply with exactly one of A, B, C, or D as your <answer>."
                ),
            }
        ],
        "answer": truth_letter,
        "example_id": example_id,
        "task": "gpqa_debate",
        "info": info,
    }


def _build_dataset(
    subset: str,
    n: int,
    seed: int,
    *,
    learner_seat_mode: str | None = None,
    learner_seat_seed: int = 0,
) -> Dataset:
    vf.ensure_keys(["HF_TOKEN"])
    raw = list(load_dataset("Idavidrein/gpqa", subset, split="train"))
    rng = random.Random(seed)
    # Separate RNG: seat-flipping shouldn't be perturbed by choice-shuffling
    # and vice versa. Seeding learner_seat_seed keeps run-to-run seat
    # assignments reproducible across reshuffles of the upstream dataset.
    seat_rng = (
        random.Random(learner_seat_seed) if learner_seat_mode == "bernoulli" else None
    )
    rows = raw if n == -1 else raw[:n]
    return Dataset.from_list(
        [
            _format_row(
                r,
                rng,
                example_idx=idx,
                learner_seat_mode=learner_seat_mode,
                seat_rng=seat_rng,
            )
            for idx, r in enumerate(rows)
        ]
    )


_SEATS = ("debater_a", "debater_b")


def _assign_learner_seat(
    *,
    example_idx: int,
    example_id: str,
    mode: str,
    rng: random.Random | None,
) -> str:
    """Pick the seat the learner occupies this episode.

    bernoulli   — coin flip from ``rng`` (per-row, stateful; seed required
                  for reproducibility)
    round_robin — ``example_idx % 2`` (deterministic, seed-free)
    hash        — ``hash(example_id) % 2`` (deterministic per example_id,
                  stable under dataset reordering — best reproducibility)
    fixed_a     — always ``debater_a``
    fixed_b     — always ``debater_b``
    """
    if mode == "bernoulli":
        if rng is None:
            raise ValueError("bernoulli mode requires a seat_rng")
        return _SEATS[int(rng.random() < 0.5)]
    if mode == "round_robin":
        return _SEATS[example_idx % 2]
    if mode == "hash":
        return _SEATS[hash(example_id) % 2]
    if mode == "fixed_a":
        return _SEATS[0]
    if mode == "fixed_b":
        return _SEATS[1]
    raise ValueError(
        f"Unknown learner_seat_mode: {mode!r} "
        f"(expected one of bernoulli, round_robin, hash, fixed_a, fixed_b)"
    )


# ---------------------------------------------------------------------------
# OpenRouter client with per-agent provider pinning
# ---------------------------------------------------------------------------


class _OpenRouterProviderClient(OpenAIChatCompletionsClient):
    """Injects OpenRouter's ``extra_body.provider`` preference on every
    ``get_response`` call. Provider prefs are per-request, so per-agent
    pinning requires one instance per distinct provider config — wired
    through ``agent_overrides={...}``.
    """

    def __init__(
        self,
        client_config: ClientConfig,
        providers: list[str] | None,
        allow_fallbacks: bool,
    ) -> None:
        super().__init__(client_config)
        self._providers = providers
        self._allow_fallbacks = allow_fallbacks

    async def get_response(
        self, prompt, model, sampling_args=None, tools=None, **kwargs
    ):
        args = dict(sampling_args or {})
        if self._providers:
            extra_body = dict(args.get("extra_body") or {})
            extra_body["provider"] = {
                "order": list(self._providers),
                "allow_fallbacks": self._allow_fallbacks,
            }
            args["extra_body"] = extra_body
        return await super().get_response(
            prompt=prompt, model=model, sampling_args=args, tools=tools, **kwargs
        )


def _make_openrouter_client(
    providers: list[str] | None, allow_fallbacks: bool
) -> _OpenRouterProviderClient:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — required for per-agent "
            "provider pinning in gpqa_debate"
        )
    return _OpenRouterProviderClient(
        ClientConfig(
            api_key_var="OPENROUTER_API_KEY",
            api_base_url="https://openrouter.ai/api/v1",
        ),
        providers=providers,
        allow_fallbacks=allow_fallbacks,
    )


# ---------------------------------------------------------------------------
# Learner-vs-fixed resolver
# ---------------------------------------------------------------------------


def _make_openai_compatible_client(
    base_url: str | None, api_key_var: str
) -> OpenAIChatCompletionsClient:
    """Plain OpenAI-compatible client (OpenAI, vLLM, SGLang, Anthropic's
    OpenAI-compatible endpoint, ...). For per-request provider pinning use
    ``_make_openrouter_client`` instead."""
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise RuntimeError(
            f"{api_key_var} is not set -- required for learner-vs-fixed "
            f"routing in gpqa_debate"
        )
    return OpenAIChatCompletionsClient(
        ClientConfig(
            api_key_var=api_key_var,
            api_base_url=base_url or "https://api.openai.com/v1",
        )
    )


def _build_learner_vs_fixed_resolver(
    *,
    opp_client: Client,
    opp_model: str,
    judge_client: Client,
    judge_model: str,
) -> Callable[[State], dict[str, tuple[Client | None, str | None]]]:
    """Closure that reads ``state.info.learner_seat`` per episode.

    Learner seat -> ``(None, None)`` so the rollout-default client/model
    (the trainer's) is used. Opposite seat -> frozen opponent endpoint.
    Judge -> frozen judge endpoint.
    """

    def resolver(state: State) -> dict[str, tuple[Client | None, str | None]]:
        learner_seat = state["info"]["learner_seat"]
        opposite_seat = "debater_b" if learner_seat == "debater_a" else "debater_a"
        return {
            learner_seat: (None, None),
            opposite_seat: (opp_client, opp_model),
            "judge": (judge_client, judge_model),
        }

    return resolver


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_environment(
    prompts_ref: str = "selfplay",
    subset: str = "gpqa_diamond",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 0,
    schedule: list[dict] | None = None,
    truth_member: str = "debater_a",
    # Self-play / OpenRouter provider-pinning path
    debater_model: str | None = None,
    judge_model: str | None = None,
    debater_providers: list[str] | None = None,
    judge_providers: list[str] | None = None,
    allow_fallbacks: bool = False,
    # Learner-vs-fixed path (mutually exclusive with the above)
    opponent_model: str | None = None,
    opponent_base_url: str | None = None,
    opponent_api_key_var: str = "OPENAI_API_KEY",
    judge_base_url: str | None = None,
    judge_api_key_var: str = "OPENAI_API_KEY",
    learner_seat_mode: str = "bernoulli",
    learner_seat_seed: int = 0,
    **extra: Any,
) -> DebateEnv:
    """vf-eval / training entry point. See module docstring for env_args schema.

    Two mutually exclusive operating modes:

    * **Self-play / OpenRouter provider pinning** — set ``debater_model``,
      ``judge_model``, and/or the ``*_providers`` kwargs. Both debater
      seats share one client; no learner/opponent asymmetry. This is the
      vf-eval benchmarking path.

    * **Learner-vs-fixed training** — set ``opponent_model`` (and
      optionally ``judge_model``). A per-episode resolver sends the
      learner seat to the rollout-default (trainer) client and the
      opposite seat to a fixed OpenAI-compatible endpoint. Seat is
      assigned from ``learner_seat_mode`` and stashed in each row's
      ``info.learner_seat``.
    """
    learner_vs_fixed = opponent_model is not None
    self_play_overrides_set = any(
        x is not None for x in (debater_model, debater_providers, judge_providers)
    )
    if learner_vs_fixed and self_play_overrides_set:
        raise ValueError(
            "gpqa_debate.load_environment: self-play kwargs "
            "(debater_model, debater_providers, judge_providers) are "
            "mutually exclusive with learner-vs-fixed (opponent_model). "
            "Pick one mode."
        )

    # Only stamp info.learner_seat when the resolver is going to read it.
    # In self-play mode the info dict stays empty -- no need to perturb
    # eval runs that never consult the key.
    seat_mode_for_dataset = learner_seat_mode if learner_vs_fixed else None

    def build_dataset() -> Dataset:
        return _build_dataset(
            subset,
            num_train_examples,
            seed,
            learner_seat_mode=seat_mode_for_dataset,
            learner_seat_seed=learner_seat_seed,
        )

    def build_eval_dataset() -> Dataset:
        # Distinct seat seed keeps eval seats independent of train seats
        # at the same example_idx (matters under round_robin / bernoulli).
        return _build_dataset(
            subset,
            num_eval_examples,
            seed + 1,
            learner_seat_mode=seat_mode_for_dataset,
            learner_seat_seed=learner_seat_seed + 1,
        )

    if learner_vs_fixed:
        opp_client = _make_openai_compatible_client(
            opponent_base_url, opponent_api_key_var
        )
        judge_client = _make_openai_compatible_client(judge_base_url, judge_api_key_var)
        resolver = _build_learner_vs_fixed_resolver(
            opp_client=opp_client,
            opp_model=opponent_model,
            judge_client=judge_client,
            # Fall back to opponent_model if no judge_model given -- common
            # case is one endpoint serving both frozen roles.
            judge_model=judge_model or opponent_model,
        )
        return _debate_load_env(
            schedule_slots=schedule or DEFAULT_SCHEDULE,
            members=["debater_a", "debater_b", "judge"],
            truth_member=truth_member,
            prompts_ref=prompts_ref,
            agent_overrides_resolver=resolver,
            dataset=build_dataset,
            eval_dataset=build_eval_dataset,
            **extra,
        )

    # Self-play / OpenRouter path -- build static agent_overrides only for
    # the dimensions the caller specified. If no per-agent overrides are
    # given, vf-eval's top-level (model, client) flows through unchanged.
    agent_overrides: dict[str, tuple] = {}

    if debater_model is not None or debater_providers is not None:
        debater_client = (
            _make_openrouter_client(debater_providers, allow_fallbacks)
            if debater_providers is not None
            else None
        )
        for mid in ("debater_a", "debater_b"):
            agent_overrides[mid] = (debater_client, debater_model)

    if judge_model is not None or judge_providers is not None:
        judge_client = (
            _make_openrouter_client(judge_providers, allow_fallbacks)
            if judge_providers is not None
            else None
        )
        agent_overrides["judge"] = (judge_client, judge_model)

    return _debate_load_env(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=prompts_ref,
        agent_overrides=agent_overrides or None,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        **extra,
    )
