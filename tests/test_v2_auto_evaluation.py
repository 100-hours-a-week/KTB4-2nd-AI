"""평가 데이터셋 정확도 자동 기록. 사진 지문 인식, 채점 규칙, 지표 보관과 실패 격리를 확인한다."""

import logging

import fakeredis.aioredis

from app.jobs.execution import execute
from app.jobs.kafka import RequestReceiver
from app.observability.metrics import Metrics
from tests.observability_helpers import SECRET, events, info_logs, rendered  # noqa: F401
from tests.test_v2_jobs import environment, message, process_request, scenario


def structured_photo(seed: int) -> bytes:
    """지문이 서로 충분히 다른 합성 사진. 큰 블록 무늬라 축소와 재압축에도 지문이 유지된다."""
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    image = Image.fromarray(np.kron(blocks, np.ones((256, 256), dtype=np.uint8))).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def be_copy(data: bytes) -> bytes:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        copy = image.convert("RGB")
        copy.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        copy.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def reference(labels):
    from app.observability.datasets import ReferenceSet, fingerprint

    hashes = tuple(fingerprint(structured_photo(i)) for i in range(len(labels)))
    return ReferenceSet("synthetic", len(labels), hashes, tuple(labels))


def test_fingerprint_survives_be_copy_and_detects_dataset():
    from app.observability.datasets import MAX_DISTANCE, detect, fingerprint

    labels = ["a"] * 6 + ["b"] * 5 + [None]
    ref = reference(labels)
    hashes = {100 + i: fingerprint(be_copy(structured_photo(i))) for i in range(len(labels))}
    assert all(ref.nearest(value)[0] <= MAX_DISTANCE for value in hashes.values())
    match = detect(hashes, sets=[ref])
    assert match.dataset == "synthetic" and match.unmatched == 0
    assert match.truth == {100 + i: label for i, label in enumerate(labels)}


def test_unrelated_trip_is_not_treated_as_evaluation():
    from app.observability.datasets import detect, fingerprint

    ref = reference(["a"] * 12)
    hashes = {i: fingerprint(be_copy(structured_photo(1000 + i))) for i in range(12)}
    # 일부만 맞으면 평가 여행으로 보지 않는다.
    hashes.update({100 + i: fingerprint(structured_photo(i)) for i in range(3)})
    assert detect(hashes, sets=[ref]) is None


def test_scoring_rules_follow_user_decisions():
    from app.observability.datasets import Match, predicted_places, score

    result = {
        "places": [
            {
                "place_id": "p1",
                "attachments": [{"trip_attachment_id": 1}, {"trip_attachment_id": 2}],
            }
        ],
        "unclassified": [
            {"trip_attachment_id": 3, "issue": "BLURRY", "place_id": "p1"},
            {"trip_attachment_id": 4, "issue": "NO_LOCATION"},
        ],
    }
    # 흐림 사진은 원래 장소, 그 밖의 미분류는 None
    assert predicted_places(result) == {1: "p1", 2: "p1", 3: "p1", 4: None}
    match = Match("synthetic", 4, {1: "a", 2: "a", 3: "a", 4: None}, unmatched=0, max_distance=1)
    reports = score(match, result, model="m", release="r")
    assert [r["unclassified_policy"] for r in reports] == ["singleton", "exclude"]
    singleton = reports[0]
    # 정답이 미분류인 사진을 미분류로 둔 것은 단독 묶음끼리 일치하므로 만점이다.
    assert singleton["metrics"]["bcubed_f1"] == 1
    assert singleton["release"] == "r" and singleton["matched_photos"] == 4
    assert reports[1]["samples"] == 3 and reports[1]["metrics"]["coverage"] == 0.75


@scenario
async def test_two_datasets_and_policies_do_not_overwrite_each_other():
    from app.observability.quality import EvaluationInput, evaluate, publish

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    items = [
        {"expected_place": "a", "predicted_place": "x"},
        {"expected_place": "b", "predicted_place": None},
    ]
    for dataset in ["jeongseon-family-trip", "eval-travel-photos"]:
        for policy in ["singleton", "exclude"]:
            report = evaluate(
                EvaluationInput(
                    feature="process",
                    dataset=dataset,
                    model="m",
                    items=items,
                    unclassified_policy=policy,
                )
            )
            await publish(redis, "test:", report)
    text = (await Metrics(redis, "test:").render()).decode()
    for dataset in ["jeongseon-family-trip", "eval-travel-photos"]:
        for policy in ["singleton", "exclude"]:
            assert (
                f'yeodam_ai_evaluation{{dataset="{dataset}",feature="process",metric="bcubed_f1",'
                f'unclassified_policy="{policy}"}}' in text
            )
    await redis.aclose()


def eval_trip(tmp_path, count=12):
    request = process_request()
    request["attachments"] = []
    for i in range(count):
        path = tmp_path / f"trips/77/{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(be_copy(structured_photo(i)))
        request["attachments"].append(
            {"trip_attachment_id": 101 + i, "analyze_storage_key": f"trips/77/{i}.jpg"}
        )
    return request


def handler_settings(monkeypatch, tmp_path, base_cfg):
    from app.embedding.client import MODEL_VERSION
    from app.jobs import process_handler
    from tests.test_process_handler import Engine

    cfg = base_cfg.model_copy(
        update={
            "IMAGE_SOURCE": "local",
            "IMAGE_DIR": tmp_path,
            "QDRANT_URL": ":memory:",
            "EMBED_MODEL": MODEL_VERSION,
            "RELEASE": "test-release",
            "JOBS_FAKE_PIPELINE": False,
        }
    )
    monkeypatch.setattr(process_handler, "JobsSettings", lambda: cfg)
    monkeypatch.setattr(process_handler, "build_engine", lambda _: Engine())
    return process_handler


@scenario
async def test_auto_evaluation_is_recorded_after_result_is_saved(caplog, monkeypatch, tmp_path):
    from app.observability import datasets
    from app.observability.metrics import QUALITY_KEY

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(datasets, "reference_sets", lambda: (reference(["a"] * 12),))
    async with environment() as (cfg, a, b, p):
        handler = handler_settings(monkeypatch, tmp_path, cfg)
        await RequestReceiver(a, p).accept(message(cfg, eval_trip(tmp_path)))
        job = await a.claim("process", "worker")
        await execute(a, job.identity, job.owner, handler.run)
        assert (await b.get(job.identity)).status == "COMPLETED"
        names = [getattr(r, "event", None) for r in caplog.records]
        assert names.index("ai_job_finished") < names.index("ai_evaluation")
        logged = events(caplog, "ai_evaluation")
        assert {r.unclassified_policy for r in logged} <= {"singleton", "exclude"} and logged
        assert all(r.dataset == "synthetic" and r.release == "test-release" for r in logged)
        quality = await a.redis.hgetall(a.key(QUALITY_KEY))
        assert any(
            k.startswith("process|synthetic|") and k.endswith("|matched_photos") for k in quality
        )
        # 이전 버전 렌더러가 읽는 키는 건드리지 않는다.
        assert not await a.redis.exists(a.key("metrics:quality"))
        stage_log = next(r for r in events(caplog, "ai_stage") if r.stage == "finalizing.evaluate")
        assert stage_log.match_ratio == 1.0 and stage_log.dataset == "synthetic"


@scenario
async def test_canceled_job_does_not_record_evaluation(caplog, monkeypatch, tmp_path):
    from app.observability import datasets

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(datasets, "reference_sets", lambda: (reference(["a"] * 12),))
    async with environment() as (cfg, a, b, p):
        handler = handler_settings(monkeypatch, tmp_path, cfg)
        await RequestReceiver(a, p).accept(message(cfg, eval_trip(tmp_path)))
        job = await a.claim("process", "worker")

        async def cancel_after(request, reporter):
            result = await handler.run(request, reporter)
            await a.cancel(77, "process")
            return result

        await execute(a, job.identity, job.owner, cancel_after)
        assert (await b.get(job.identity)).status == "CANCELED"
        assert not events(caplog, "ai_evaluation")
        assert events(caplog, "ai_evaluation_skipped")[0].result == "CANCELED"


@scenario
async def test_broken_reference_files_only_disable_evaluation(caplog, monkeypatch, tmp_path):
    from app.observability import datasets

    caplog.set_level(logging.INFO)

    def broken():
        raise ValueError("지문 방식 불일치")

    monkeypatch.setattr(datasets, "reference_sets", broken)
    async with environment() as (cfg, a, b, p):
        handler = handler_settings(monkeypatch, tmp_path, cfg)
        await RequestReceiver(a, p).accept(message(cfg, eval_trip(tmp_path)))
        job = await a.claim("process", "worker")
        await execute(a, job.identity, job.owner, handler.run)
        assert (await b.get(job.identity)).status == "COMPLETED"
        assert events(caplog, "ai_evaluation_failed")[0].error_type == "ValueError"


def test_damaged_exif_never_raises_from_fingerprint():
    import io
    import struct

    from PIL import Image

    from app.observability.datasets import candidates, fingerprint

    cases = [
        b"Exif\x00\x00XXXXXXXX",
        b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x05\x01\x12",
        b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08"
        + struct.pack(">H", 2)
        + struct.pack(">HHI", 0x0112, 3, 1)
        + struct.pack(">HH", 6, 0)
        + struct.pack(">HHI", 0x010F, 2, 200)
        + struct.pack(">I", 0xFFFFFF00)
        + b"\x00\x00\x00\x00",
    ]
    for exif in cases:
        for dpi in (None, (72, 72)):
            buffer = io.BytesIO()
            options = {"quality": 85, "exif": exif, **({"dpi": dpi} if dpi else {})}
            Image.new("RGB", (64, 48), (120, 30, 200)).save(buffer, "JPEG", **options)
            data = buffer.getvalue()
            fingerprint(data)
            assert candidates(data) is None or all(isinstance(v, int) for v in candidates(data))


def test_rotated_copy_that_keeps_orientation_tag_is_still_recognized():
    import io

    from PIL import Image, ImageOps

    from app.observability.datasets import ReferenceSet, candidates, detect, fingerprint

    labels = ["a"] * 12
    originals = []
    for i in range(len(labels)):
        image = Image.open(io.BytesIO(structured_photo(i))).convert("RGB")
        exif = image.getexif()
        exif[0x0112] = 6  # 90도 회전해서 보라는 방향 태그
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=95, exif=exif.tobytes())
        originals.append(buffer.getvalue())
    ref = ReferenceSet("rotated", 12, tuple(fingerprint(d) for d in originals), tuple(labels))
    for variant in ["rotate_and_drop_tag", "keep_pixels_and_tag", "rotate_and_keep_tag"]:
        hashes = {}
        for i, data in enumerate(originals):
            with Image.open(io.BytesIO(data)) as src:
                exif = src.getexif()
                pixels = src if variant == "keep_pixels_and_tag" else ImageOps.exif_transpose(src)
                copy = pixels.convert("RGB")
                copy.thumbnail((1024, 1024))
                out = io.BytesIO()
                keep = variant != "rotate_and_drop_tag"
                copy.save(out, "JPEG", quality=85, **({"exif": exif.tobytes()} if keep else {}))
            hashes[i] = candidates(out.getvalue())
        assert detect(hashes, sets=[ref]) is not None, variant


def test_exclude_policy_is_skipped_only_when_every_photo_is_unclassified():
    from app.observability.datasets import Match, score

    result = {"places": [], "unclassified": [{"trip_attachment_id": 1, "issue": "NO_LOCATION"}]}
    match = Match("synthetic", 1, {1: "a"}, unmatched=0, max_distance=0)
    reports = score(match, result, model="m", release="r")
    assert [r["unclassified_policy"] for r in reports] == ["singleton"]


@scenario
async def test_auto_evaluation_failure_never_fails_processing(caplog, monkeypatch):
    from app.jobs import process_handler
    from app.jobs.config import JobsSettings
    from app.observability import datasets

    def broken(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(datasets, "closest", broken)
    cfg = JobsSettings(_env_file=None, API_KEY="test", IMAGE_SOURCE="local")
    assert process_handler.auto_evaluate({"places": [], "unclassified": []}, {1: 1}, cfg) == []
    assert events(caplog, "ai_evaluation_failed")[0].error_type == "RuntimeError"
    assert SECRET not in rendered(caplog)


@scenario
async def test_partial_upload_is_logged_but_not_published(caplog, monkeypatch, tmp_path):
    from app.observability import datasets
    from app.observability.metrics import QUALITY_KEY

    caplog.set_level(logging.INFO)
    # 기준 데이터셋은 20장인데 그중 12장만 올린 여행
    monkeypatch.setattr(datasets, "reference_sets", lambda: (reference(["a"] * 20),))
    async with environment() as (cfg, a, b, p):
        handler = handler_settings(monkeypatch, tmp_path, cfg)
        await RequestReceiver(a, p).accept(message(cfg, eval_trip(tmp_path, count=12)))
        job = await a.claim("process", "worker")
        await execute(a, job.identity, job.owner, handler.run)
        logged = events(caplog, "ai_evaluation")
        assert logged and all(r.dashboard is False and r.dataset_coverage == 0.6 for r in logged)
        assert not await a.redis.exists(a.key(QUALITY_KEY))
