"""실제 가중치 없이 임베딩 엔진의 배치 처리와 출력 계약을 검증한다."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from app.engines import siglip_naflex
from app.engines.siglip_naflex import SiglipNaflexEngine


def test_loads_local_vision_model_in_bfloat16(monkeypatch: pytest.MonkeyPatch, tmp_path):
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    load_model = Mock(return_value=model)
    processor = object()
    load_processor = Mock(return_value=processor)
    monkeypatch.setattr(siglip_naflex.Siglip2VisionModel, "from_pretrained", load_model)
    monkeypatch.setattr(siglip_naflex.AutoImageProcessor, "from_pretrained", load_processor)

    subject = SiglipNaflexEngine(tmp_path, device="cpu", batch_size=2)

    load_model.assert_called_once_with(tmp_path, local_files_only=True, dtype=torch.bfloat16)
    load_processor.assert_called_once_with(tmp_path, local_files_only=True)
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()
    assert subject.model is model
    assert subject.processor is processor
    assert subject.batch_size == 2


@pytest.fixture(params=[torch.float64, torch.bfloat16], ids=["float64", "bfloat16"])
def engine(monkeypatch: pytest.MonkeyPatch, request):
    # 사진별로 서로 다른 방향과 길이의 벡터를 반환한다.
    raw_vectors = torch.zeros((3, 1152), dtype=request.param)
    raw_vectors[0, :2] = torch.tensor([3.0, 4.0])
    raw_vectors[1, :2] = torch.tensor([0.0, 2.0])
    raw_vectors[2, :2] = torch.tensor([-4.0, 3.0])
    batches = []

    class Processor:
        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            ids = [image.getpixel((0, 0))[0] for image in images]
            batches.append(ids)
            return BatchFeature({"image_ids": torch.tensor(ids)})

    class Model:
        def __call__(self, image_ids):
            assert torch.is_inference_mode_enabled()
            return SimpleNamespace(pooler_output=raw_vectors[image_ids])

    def fake_init(self):
        self.device = "cpu"
        self.batch_size = 2
        self.processor = Processor()
        self.model = Model()

    monkeypatch.setattr(SiglipNaflexEngine, "__init__", fake_init)
    return SiglipNaflexEngine(), batches


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_encode_images_preserves_order_and_normalizes_across_batches(engine, batch_size):
    subject, batches = engine
    subject.batch_size = batch_size
    images = [Image.new("RGB", (2, 2), (i, 0, 0)) for i in [2, 0, 1]]

    vectors = subject.encode_images(images)

    expected = np.zeros((3, 1152), dtype=np.float32)
    expected[:, :2] = [[-0.8, 0.6], [0.6, 0.8], [0.0, 1.0]]
    assert vectors.shape == (3, 1152)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(vectors, expected, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-7)
    assert [i for batch in batches for i in batch] == [2, 0, 1]
    assert all(0 < len(batch) <= batch_size for batch in batches)


def test_encode_images_empty_input_skips_processing(engine):
    subject, batches = engine

    vectors = subject.encode_images([])

    assert vectors.shape == (0, 1152)
    assert vectors.dtype == np.float32
    assert batches == []
