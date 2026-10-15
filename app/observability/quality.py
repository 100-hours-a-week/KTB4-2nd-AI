"""정답을 포함한 평가 입력으로 품질을 계산한다. 운영 응답을 정답으로 취급하지 않는다."""

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import adjusted_rand_score


class EvaluationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: Literal["process", "search", "story"]
    dataset: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    items: list[dict] = Field(min_length=1)
    unclassified_policy: Literal["singleton", "exclude"] | None = None


def evaluate(data: EvaluationInput) -> dict:
    total_samples = len(data.items)
    coverage = None
    if data.feature == "process":
        missing = sum(item["predicted_place"] is None for item in data.items)
        coverage = 1 - missing / total_samples
        if missing and data.unclassified_policy is None:
            raise ValueError("미분류 평가 방식은 singleton 또는 exclude로 명시해야 합니다.")
        if data.unclassified_policy == "exclude":
            data = data.model_copy(
                update={
                    "items": [item for item in data.items if item["predicted_place"] is not None]
                }
            )
            if not data.items:
                raise ValueError("미분류 제외 후 평가할 사진이 없습니다.")
        truth = [item["expected_place"] for item in data.items]
        predicted = [
            item["predicted_place"] if item["predicted_place"] is not None else ("unclassified", i)
            for i, item in enumerate(data.items)
        ]
        if any(not isinstance(label, str) or not label for label in truth):
            raise ValueError("정리 정답 장소는 비어 있지 않은 문자열이어야 합니다.")
        if any(
            item["predicted_place"] is not None and not isinstance(item["predicted_place"], str)
            for item in data.items
        ):
            raise ValueError("예측 장소는 문자열 또는 null이어야 합니다.")
        # 정답과 예측의 이름은 서로 같을 필요가 없다. 각 묶음의 교집합으로 계산한다.
        a, b, pairs = (
            Counter(truth),
            Counter(predicted),
            Counter(zip(truth, predicted, strict=True)),
        )
        precision = sum(pairs[t, p] / b[p] for t, p in zip(truth, predicted, strict=True)) / len(
            truth
        )
        recall = sum(pairs[t, p] / a[t] for t, p in zip(truth, predicted, strict=True)) / len(truth)
        indices = {label: i for i, label in enumerate(set(predicted))}
        scores = {
            "bcubed_precision": precision,
            "bcubed_recall": recall,
            "bcubed_f1": 2 * precision * recall / (precision + recall),
            "ari": float(adjusted_rand_score(truth, [indices[p] for p in predicted])),
            "coverage": coverage,
        }
    elif data.feature == "search":
        if len({item["k"] for item in data.items}) != 1:
            raise ValueError("하나의 검색 평가에서는 같은 k를 사용해야 합니다.")
        values = []
        for item in data.items:
            k = item["k"]
            expected = item["relevant_ids"]
            retrieved = item["retrieved_ids"]
            if isinstance(k, bool) or not isinstance(k, int) or k < 1 or not expected:
                raise ValueError("검색 평가는 양의 k와 정답 사진이 필요합니다.")
            if len(set(expected)) != len(expected) or len(set(retrieved)) != len(retrieved):
                raise ValueError("평가 사진 ID를 중복할 수 없습니다.")
            found = [i + 1 for i, photo in enumerate(retrieved[:k]) if photo in set(expected)]
            values.append(
                (len(found) / k, len(found) / len(expected), 1 / found[0] if found else 0)
            )
        scores = {
            name: sum(row[i] for row in values) / len(values)
            for i, name in enumerate(["precision_at_k", "recall_at_k", "mrr_at_k"])
        }
    else:
        fields = ["factual", "all_places_present", "mood_matched"]
        if any(type(item[field]) is not bool for item in data.items for field in fields):
            raise ValueError("스토리 품질은 사람이 판정한 boolean 정답이 필요합니다.")
        scores = {
            field + "_rate": sum(item[field] for item in data.items) / len(data.items)
            for field in fields
        }
    return {
        "event": "ai_evaluation",
        "feature": data.feature,
        "dataset": data.dataset,
        "model": data.model,
        "samples": len(data.items),
        "total_samples": total_samples,
        "unclassified_policy": data.unclassified_policy,
        "k": data.items[0]["k"] if data.feature == "search" else None,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": scores,
    }


async def publish(redis, prefix, report):
    feature = report["feature"]
    mapping = {feature + "|" + name: value for name, value in report["metrics"].items()}
    mapping[feature + "|samples"] = report["samples"]
    mapping[feature + "|timestamp"] = datetime.fromisoformat(report["evaluated_at"]).timestamp()
    metadata = {name: report[name] for name in ["dataset", "model", "k", "unclassified_policy"]}
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(prefix + "metrics:quality", mapping=mapping)
        pipe.hset(prefix + "metrics:quality_context", feature, json.dumps(metadata))
        await pipe.execute()
