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

    # Self-play / OpenRouter bindings (each optional; mutually exclusive
    # with the external-opponent kwargs below)
    debater_model      str      slug for both debaters                 None (use default)
    judge_model        str      slug for the judge                     None (use default)
    debater_providers  list[str]  strict provider order for debaters   None (OR chooses)
    judge_providers    list[str]  strict provider order for the judge  None
    allow_fallbacks    bool     if true, OR may fall back to non-listed  False

    # External opponent (set opponent_model to enable)
    opponent_model       str   external-opponent model id              None (self-play)
    opponent_base_url    str   OpenAI-compatible endpoint              None (api.openai.com)
    opponent_api_key_var str   env var holding opponent API key        "OPENAI_API_KEY"
    judge_base_url       str   judge endpoint (fallback: opponent)     None (api.openai.com)
    judge_api_key_var    str   env var holding judge API key           "OPENAI_API_KEY"
    learner_seat_mode    str   bernoulli | round_robin | hash          "bernoulli"
    learner_seat_seed    int   RNG seed for bernoulli mode             0
    pin_learner_seat     str   "a" | "b" — hard override (ablation)    None
"""

from __future__ import annotations

import hashlib
import logging
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

_log = logging.getLogger(__name__)

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
    seat_mode: str | None,
    pin: str | None,
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
    if seat_mode is not None:
        info["learner_seat"] = _pick_learner_seat(
            example_idx=example_idx,
            example_id=example_id,
            mode=seat_mode,
            pin=pin,
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
    pin_learner_seat: str | None = None,
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
                seat_mode=learner_seat_mode,
                pin=pin_learner_seat,
                seat_rng=seat_rng,
            )
            for idx, r in enumerate(rows)
        ]
    )


_SEATS = ("debater_a", "debater_b")
_SEAT_BY_LETTER = {"a": "debater_a", "b": "debater_b"}


def _pick_learner_seat(
    *,
    example_idx: int,
    example_id: str,
    mode: str,
    pin: str | None,
    rng: random.Random | None,
) -> str:
    """Pick the seat the learner occupies this episode.

    ``pin`` is the hard override: when set, seat is ``debater_<pin>`` and
    ``mode`` is ignored. Use it for ablations (e.g. "always side A")
    and for curriculum experiments that vary which side the learner
    defends.

    Otherwise ``mode`` selects a seat-distribution mechanism. All three
    try to balance seats across examples; they differ in determinism:

    bernoulli   — coin flip from ``rng`` (stateful; requires seed for
                  reproducibility)
    round_robin — ``example_idx % 2`` (deterministic, seed-free)
    hash        — ``hash(example_id) % 2`` (deterministic per example_id,
                  stable under dataset reordering)
    """
    if pin is not None:
        seat = _SEAT_BY_LETTER.get(pin)
        if seat is None:
            raise ValueError(f"pin_learner_seat must be 'a' or 'b', got {pin!r}")
        return seat
    if mode == "bernoulli":
        if rng is None:
            raise ValueError("bernoulli mode requires a seat_rng")
        return _SEATS[int(rng.random() < 0.5)]
    if mode == "round_robin":
        return _SEATS[example_idx % 2]
    if mode == "hash":
        # Python's built-in hash() is process-randomized (PYTHONHASHSEED),
        # which defeats this mode's "deterministic / stable across runs"
        # contract. blake2b is stable and fast.
        digest = hashlib.blake2b(example_id.encode("utf-8"), digest_size=1).digest()
        return _SEATS[digest[0] & 1]
    raise ValueError(
        f"Unknown learner_seat_mode: {mode!r} "
        f"(expected one of bernoulli, round_robin, hash)"
    )


# ---------------------------------------------------------------------------
# OpenRouter client with per-agent provider pinning
# ---------------------------------------------------------------------------


class _OpenRouterProviderClient(OpenAIChatCompletionsClient):
    """Injects OpenRouter's ``extra_body.provider`` preference on every
    ``get_response`` call. Provider prefs are per-request, so per-agent
    pinning requires one instance per distinct provider config — wired
    through ``agent_bindings={...}``.
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
# External-opponent bindings
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
            f"{api_key_var} is not set -- required for external-opponent "
            f"routing in gpqa_debate"
        )
    return OpenAIChatCompletionsClient(
        ClientConfig(
            api_key_var=api_key_var,
            api_base_url=base_url or "https://api.openai.com/v1",
        )
    )


def _build_external_opponent_bindings(
    *,
    opp_client: Client | None,
    opp_model: str,
    judge_client: Client | None,
    judge_model: str,
) -> Callable[[State], dict[str, tuple[Client | None, str | None]]]:
    """Return an ``agent_bindings_fn`` that reads ``info.learner_seat``
    per episode and binds:

      learner seat  -> (None, None)           rollout-default = trainer's client
      opposite seat -> (opp_client,   opp_model)
      judge         -> (judge_client, judge_model)

    ``opp_client`` / ``judge_client`` may be ``None``, in which case the
    rollout-default client is reused for that seat -- only the model
    string differs. That enables the shared-vLLM / LoRA-disabled
    opponent pattern: launch vLLM with ``--enable-lora --lora-modules
    learner_adapter=<path>``, have the trainer sync the adapter, route
    the learner seat to the adapter alias (via rollout-default) and the
    opposite seat to the base model name (same client, same server, no
    adapter applied).

    The caller is responsible for the judge_client fallback policy --
    typically "inherit opponent's client when no explicit judge
    endpoint is set" so the judge lands on the frozen endpoint rather
    than silently routing through the learner's.
    """

    def bindings_fn(state: State) -> dict[str, tuple[Client | None, str | None]]:
        learner_seat = state["info"]["learner_seat"]
        opposite_seat = "debater_b" if learner_seat == "debater_a" else "debater_a"
        return {
            learner_seat: (None, None),
            opposite_seat: (opp_client, opp_model),
            "judge": (judge_client, judge_model),
        }

    return bindings_fn


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
    # External-opponent path (mutually exclusive with the above)
    opponent_model: str | None = None,
    opponent_base_url: str | None = None,
    opponent_api_key_var: str = "OPENAI_API_KEY",
    judge_base_url: str | None = None,
    judge_api_key_var: str = "OPENAI_API_KEY",
    learner_seat_mode: str = "bernoulli",
    learner_seat_seed: int = 0,
    pin_learner_seat: str | None = None,
    **extra: Any,
) -> DebateEnv:
    """vf-eval / training entry point. See module docstring for env_args schema.

    Two mutually exclusive operating modes:

    * **Self-play / OpenRouter provider pinning** — set ``debater_model``,
      ``judge_model``, and/or the ``*_providers`` kwargs. Both debater
      seats share one client; no learner/opponent asymmetry. This is the
      vf-eval benchmarking path.

    * **External opponent** — set ``opponent_model`` (and optionally
      ``judge_model``). A per-episode ``agent_bindings_fn`` sends the
      learner seat to the rollout-default (trainer) client and the
      opposite seat to a fixed OpenAI-compatible endpoint. Seat is
      assigned per-row by ``learner_seat_mode`` (or ``pin_learner_seat``
      for ablations) and stashed in each row's ``info.learner_seat``.
    """
    external_opponent = opponent_model is not None
    self_play_overrides_set = any(
        x is not None for x in (debater_model, debater_providers, judge_providers)
    )
    if external_opponent and self_play_overrides_set:
        raise ValueError(
            "gpqa_debate.load_environment: self-play kwargs "
            "(debater_model, debater_providers, judge_providers) are "
            "mutually exclusive with external-opponent (opponent_model). "
            "Pick one mode."
        )

    # Only stamp info.learner_seat when the bindings fn is going to read
    # it. In self-play mode the info dict stays empty -- no need to
    # perturb eval runs that never consult the key.
    seat_mode_for_dataset = learner_seat_mode if external_opponent else None
    pin_for_dataset = pin_learner_seat if external_opponent else None

    def build_dataset() -> Dataset:
        return _build_dataset(
            subset,
            num_train_examples,
            seed,
            learner_seat_mode=seat_mode_for_dataset,
            learner_seat_seed=learner_seat_seed,
            pin_learner_seat=pin_for_dataset,
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
            pin_learner_seat=pin_for_dataset,
        )

    if external_opponent:
        # Client routing is three-way:
        #   * base_url set      -> dedicated client for that endpoint
        #   * base_url is None  -> None (= reuse rollout-default client,
        #                          i.e. same vLLM as the learner, handy
        #                          for the LoRA-self setup)
        # Judge gets one extra fallback on top: if no judge endpoint was
        # given but an opponent endpoint was, reuse the opponent client --
        # the common-case deployment is one frozen endpoint serving both
        # non-learner roles, which is exactly what the judge_model fallback
        # to opponent_model encodes. Without this tier, judge traffic
        # would silently route to the learner's endpoint whenever the
        # caller relied on the documented "judge falls back to opponent"
        # path.
        opp_client = (
            _make_openai_compatible_client(opponent_base_url, opponent_api_key_var)
            if opponent_base_url is not None
            else None
        )
        if judge_base_url is not None:
            judge_client = _make_openai_compatible_client(
                judge_base_url, judge_api_key_var
            )
        elif opponent_base_url is not None:
            judge_client = opp_client
        else:
            judge_client = None

        if opponent_base_url is None:
            _log.warning(
                "gpqa_debate: opponent_base_url not set; opponent (and judge) "
                "will share the rollout-default client (same vLLM as the "
                "learner). This is the shared-vLLM / LoRA-self topology -- "
                "opponent routes to model=%r on the learner's endpoint. For "
                "the learner and opponent to produce different outputs, "
                "launch vLLM with --enable-lora and have the trainer sync "
                "the learner's adapter under a distinct model alias; "
                "otherwise both seats will produce identical rollouts "
                "(degenerate self-play). Run "
                "scripts/preflight_lora_smoke.py before the first training "
                "step. Set opponent_base_url to a distinct endpoint to opt "
                "into the two-instance topology instead.",
                opponent_model,
            )
        bindings_fn = _build_external_opponent_bindings(
            opp_client=opp_client,
            opp_model=opponent_model,
            judge_client=judge_client,
            # Fall back to opponent_model if no judge_model given -- common
            # case is one endpoint serving both non-learner roles.
            judge_model=judge_model or opponent_model,
        )
        return _debate_load_env(
            schedule_slots=schedule or DEFAULT_SCHEDULE,
            members=["debater_a", "debater_b", "judge"],
            truth_member=truth_member,
            prompts_ref=prompts_ref,
            agent_bindings_fn=bindings_fn,
            dataset=build_dataset,
            eval_dataset=build_eval_dataset,
            **extra,
        )

    # Self-play / OpenRouter path -- build static agent_bindings only for
    # the dimensions the caller specified. If no per-agent bindings are
    # given, vf-eval's top-level (model, client) flows through unchanged.
    agent_bindings: dict[str, tuple] = {}

    if debater_model is not None or debater_providers is not None:
        debater_client = (
            _make_openrouter_client(debater_providers, allow_fallbacks)
            if debater_providers is not None
            else None
        )
        for mid in ("debater_a", "debater_b"):
            agent_bindings[mid] = (debater_client, debater_model)

    if judge_model is not None or judge_providers is not None:
        judge_client = (
            _make_openrouter_client(judge_providers, allow_fallbacks)
            if judge_providers is not None
            else None
        )
        agent_bindings["judge"] = (judge_client, judge_model)

    return _debate_load_env(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=prompts_ref,
        agent_bindings=agent_bindings or None,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        **extra,
    )
