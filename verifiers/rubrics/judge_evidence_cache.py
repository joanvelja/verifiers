from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JudgeEvidenceSample:
    raw_response: str
    verdict: str


@dataclass(frozen=True)
class JudgeDecision:
    p_correct: float
    reward: float
    hard_response: str
    raw_response: str
    n_valid: int
    n_effective: float
    support: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_correct": self.p_correct,
            "reward": self.reward,
            "hard_response": self.hard_response,
            "raw_response": self.raw_response,
            "n_valid": self.n_valid,
            "n_effective": self.n_effective,
            "support": self.support,
        }


@dataclass(frozen=True)
class JudgePanelPolicy:
    positive_label: str = "YES"
    negative_label: str = "NO"
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    threshold: float = 0.5
    reward_mode: str = "hard"
    repeated_call_correlation: float = 0.0
    calibration_mode: str = "vote_fraction"
    correctness_prior: float = 0.5
    judge_sensitivity: float | None = None
    judge_false_positive_rate: float | None = None

    def __post_init__(self) -> None:
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("JudgePanelPolicy priors must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("JudgePanelPolicy threshold must be in [0, 1]")
        if not 0.0 <= self.repeated_call_correlation <= 1.0:
            raise ValueError(
                "JudgePanelPolicy repeated_call_correlation must be in [0, 1]"
            )
        if self.reward_mode not in {"hard", "soft", "centered"}:
            raise ValueError(
                "JudgePanelPolicy reward_mode must be hard, soft, or centered"
            )
        if self.calibration_mode not in {"vote_fraction", "confusion_matrix"}:
            raise ValueError(
                "JudgePanelPolicy calibration_mode must be vote_fraction "
                "or confusion_matrix"
            )
        if not 0.0 <= self.correctness_prior <= 1.0:
            raise ValueError("JudgePanelPolicy correctness_prior must be in [0, 1]")
        if self.calibration_mode == "confusion_matrix":
            if self.judge_sensitivity is None or self.judge_false_positive_rate is None:
                raise ValueError(
                    "JudgePanelPolicy confusion_matrix mode requires "
                    "judge_sensitivity and judge_false_positive_rate"
                )
            if not 0.0 <= self.judge_sensitivity <= 1.0:
                raise ValueError("JudgePanelPolicy judge_sensitivity must be in [0, 1]")
            if not 0.0 <= self.judge_false_positive_rate <= 1.0:
                raise ValueError(
                    "JudgePanelPolicy judge_false_positive_rate must be in [0, 1]"
                )

    @property
    def positive_token(self) -> str:
        return verdict_token(self.positive_label)

    @property
    def negative_token(self) -> str:
        return verdict_token(self.negative_label)

    @property
    def positive_tokens(self) -> set[str]:
        tokens = {self.positive_token}
        if self.positive_token in {"YES", "CORRECT"}:
            tokens.update({"YES", "CORRECT"})
        return tokens

    @property
    def negative_tokens(self) -> set[str]:
        tokens = {self.negative_token}
        if self.negative_token in {"NO", "INCORRECT"}:
            tokens.update({"NO", "INCORRECT"})
        return tokens

    def decide(self, samples: list[JudgeEvidenceSample]) -> JudgeDecision:
        valid = [
            sample
            for sample in samples
            if verdict_token(sample.verdict)
            in self.positive_tokens | self.negative_tokens
        ]
        n_valid = len(valid)
        if n_valid == 0:
            p_correct = self.prior_alpha / (self.prior_alpha + self.prior_beta)
            hard_response = (
                self.positive_token
                if p_correct > self.threshold
                else self.negative_token
            )
            return JudgeDecision(
                p_correct=p_correct,
                reward=self._reward(p_correct),
                hard_response=hard_response,
                raw_response="",
                n_valid=0,
                n_effective=0.0,
                support={
                    "positive": 0,
                    "negative": 0,
                    "invalid": len(samples),
                    "policy": self._support_policy(),
                },
            )

        positives = sum(
            1
            for sample in valid
            if verdict_token(sample.verdict) in self.positive_tokens
        )
        negatives = n_valid - positives
        n_effective = n_valid / (1.0 + (n_valid - 1) * self.repeated_call_correlation)
        sample_weight = n_effective / n_valid
        p_correct = self._p_correct(
            positives=positives,
            negatives=negatives,
            sample_weight=sample_weight,
            n_effective=n_effective,
        )
        hard_tokens = (
            self.positive_tokens if p_correct > self.threshold else self.negative_tokens
        )
        raw_response = self._representative_raw_response(valid, hard_tokens)
        hard_response = verdict_token(raw_response) or (
            self.positive_token if p_correct > self.threshold else self.negative_token
        )
        return JudgeDecision(
            p_correct=p_correct,
            reward=self._reward(p_correct),
            hard_response=hard_response,
            raw_response=raw_response,
            n_valid=n_valid,
            n_effective=n_effective,
            support={
                "positive": positives,
                "negative": negatives,
                "invalid": len(samples) - n_valid,
                "policy": self._support_policy(),
            },
        )

    def _p_correct(
        self,
        *,
        positives: int,
        negatives: int,
        sample_weight: float,
        n_effective: float,
    ) -> float:
        if self.calibration_mode == "confusion_matrix":
            return self._confusion_matrix_p_correct(
                positives=positives,
                negatives=negatives,
                sample_weight=sample_weight,
            )
        weighted_positive = positives * sample_weight
        return (self.prior_alpha + weighted_positive) / (
            self.prior_alpha + self.prior_beta + n_effective
        )

    def _confusion_matrix_p_correct(
        self,
        *,
        positives: int,
        negatives: int,
        sample_weight: float,
    ) -> float:
        prior = self._clip_probability(self.correctness_prior)
        sensitivity = self._clip_probability(self.judge_sensitivity)
        false_positive_rate = self._clip_probability(self.judge_false_positive_rate)

        log_odds = math.log(prior / (1.0 - prior))
        log_odds += (
            positives * sample_weight * math.log(sensitivity / false_positive_rate)
        )
        log_odds += (
            negatives
            * sample_weight
            * math.log((1.0 - sensitivity) / (1.0 - false_positive_rate))
        )
        return self._sigmoid(log_odds)

    @staticmethod
    def _clip_probability(value: float | None) -> float:
        if value is None:
            raise ValueError("Probability value is required")
        return min(max(float(value), 1e-12), 1.0 - 1e-12)

    @staticmethod
    def _sigmoid(log_odds: float) -> float:
        if log_odds >= 0:
            z = math.exp(-log_odds)
            return 1.0 / (1.0 + z)
        z = math.exp(log_odds)
        return z / (1.0 + z)

    def _reward(self, p_correct: float) -> float:
        if self.reward_mode == "soft":
            return p_correct
        if self.reward_mode == "centered":
            return 2 * p_correct - 1
        return 1.0 if p_correct > self.threshold else 0.0

    def _support_policy(self) -> dict[str, Any]:
        return {
            "positive_label": self.positive_token,
            "negative_label": self.negative_token,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "threshold": self.threshold,
            "reward_mode": self.reward_mode,
            "repeated_call_correlation": self.repeated_call_correlation,
            "calibration_mode": self.calibration_mode,
            "correctness_prior": self.correctness_prior,
            "judge_sensitivity": self.judge_sensitivity,
            "judge_false_positive_rate": self.judge_false_positive_rate,
        }

    @staticmethod
    def _representative_raw_response(
        samples: list[JudgeEvidenceSample], hard_tokens: set[str]
    ) -> str:
        for sample in reversed(samples):
            if verdict_token(sample.verdict) in hard_tokens:
                return sample.raw_response
        return samples[-1].raw_response


def canonical_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def verdict_token(response: str) -> str:
    text = canonical_text(response)
    if not text:
        return ""
    return re.sub(r"^\W+|\W+$", "", text.split(maxsplit=1)[0]).upper()


def stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


class SQLiteJudgeEvidenceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS judge_cells (
                    cell_key TEXT PRIMARY KEY,
                    rubric_family TEXT NOT NULL,
                    question_id TEXT,
                    question_hash TEXT NOT NULL,
                    target TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS judge_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cell_key TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    grader_model TEXT,
                    sampling_args_json TEXT,
                    prompt_hash TEXT,
                    system_prompt_hash TEXT,
                    metadata_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(cell_key) REFERENCES judge_cells(cell_key)
                );

                CREATE INDEX IF NOT EXISTS idx_judge_samples_cell_variant
                    ON judge_samples(cell_key, variant_id);
                """
            )

    def get_samples(self, cell_key: str, variant_id: str) -> list[JudgeEvidenceSample]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT raw_response, verdict
                FROM judge_samples
                WHERE cell_key = ? AND variant_id = ?
                ORDER BY id ASC
                """,
                (cell_key, variant_id),
            ).fetchall()
        return [
            JudgeEvidenceSample(
                raw_response=str(row["raw_response"]), verdict=str(row["verdict"])
            )
            for row in rows
        ]

    def add_sample(
        self,
        *,
        cell_key: str,
        identity: dict[str, str | None],
        variant_id: str,
        raw_response: str,
        grader_model: str,
        sampling_args: dict[str, Any],
        judge_prompt: str,
        judge_system_prompt: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO judge_cells (
                    cell_key, rubric_family, question_id, question_hash, target,
                    response, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cell_key) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (
                    cell_key,
                    identity["rubric_family"],
                    identity.get("question_id"),
                    identity["question_hash"],
                    identity["target"],
                    identity["response"],
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO judge_samples (
                    cell_key, variant_id, verdict, raw_response, grader_model,
                    sampling_args_json, prompt_hash, system_prompt_hash,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cell_key,
                    variant_id,
                    verdict_token(raw_response),
                    raw_response,
                    grader_model,
                    json.dumps(
                        sampling_args,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=repr,
                    ),
                    stable_hash({"prompt": judge_prompt}),
                    stable_hash({"system": judge_system_prompt})
                    if judge_system_prompt
                    else None,
                    json.dumps(
                        metadata or {},
                        sort_keys=True,
                        separators=(",", ":"),
                        default=repr,
                    ),
                    now,
                ),
            )
