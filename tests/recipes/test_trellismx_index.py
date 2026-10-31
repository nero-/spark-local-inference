# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from ignition.recipes import _replacement, _stored_bytes, _supporting
from ignition.recipes.deepseek41.decoder import PreparedTrellisExperts
from ignition.recipes.deepseek41.engram import Tables, _tensor_span


def test_trellismx_stored_bytes_sums_snapshot(tmp_path):
    (tmp_path / "trellismx").mkdir()
    (tmp_path / "trellismx" / "manifest.json").write_text("{}")
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 32)
    (tmp_path / "engram").mkdir()
    (tmp_path / "engram" / "layer-001.trellis.safetensors").write_bytes(b"y" * 64)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 32}, "weight_map": {}}))

    assert _stored_bytes(index, "test") == sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())


def test_stored_bytes_uses_index_total_for_non_trellis(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 1234}, "weight_map": {}}))

    assert _stored_bytes(index, "test") == 1234


def test_trellismx_stored_bytes_ignores_download_caches(tmp_path):
    (tmp_path / "trellismx").mkdir()
    (tmp_path / "trellismx" / "manifest.json").write_text("{}")
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 32)
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "blob").write_bytes(b"y" * 4096)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "lfs").write_bytes(b"z" * 4096)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 32}, "weight_map": {}}))

    expected = 32 + index.stat().st_size + (tmp_path / "trellismx" / "manifest.json").stat().st_size
    assert _stored_bytes(index, "test") == expected


def test_tensor_span_exposes_a_row_width_mismatch(tmp_path):
    def _write(path, end):
        header = json.dumps({"layers.1.engram.embed.trellis": {
            "dtype": "U8", "shape": [4, 128], "data_offsets": [0, end]}}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * end)
        return 8 + len(header)

    good = tmp_path / "layer-001.trellis.safetensors"
    _write(good, 512)
    offset, nbytes, shape, dtype = _tensor_span(good, "layers.1.engram.embed.trellis")
    assert nbytes == 512 and shape == (4, 128) and dtype == "U8" and offset >= 8

    short = tmp_path / "layer-014.trellis.safetensors"
    _write(short, 256)
    offset, nbytes, shape, dtype = _tensor_span(short, "layers.1.engram.embed.trellis")
    assert nbytes == 256 and shape == (4, 128)  # caller rejects nbytes != rows*width


def test_supporting_selects_the_trellismx_checkpoint_by_quant_method(tmp_path):
    trellis = tmp_path / "trellis.json"
    trellis.write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"],
                                   "quantization_config": {"quant_method": "ds41_trellismx"}}))
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"],
                                    "quantization_config": {"quant_method": "fp8"}}))

    _, trained = _supporting(trellis, "trellis")
    _, stock = _supporting(official, "official")
    assert "trellismx" in trained.alias and "trellismx" not in stock.alias
    # The pack's per-rank footprint must not shrink with a rank-0-only subset.
    kept = _replacement(trained, "dir", "rev", trained.stored_bytes // 2, scale=False)
    assert kept.device_bytes == trained.device_bytes


def test_trellis_experts_follow_the_functional_protocol():
    source = torch.zeros(2, 16, dtype=torch.bfloat16)
    ids = torch.zeros(2, 6, dtype=torch.int32)
    weights = torch.ones(2, 6)
    output = torch.zeros(2, 16, dtype=torch.bfloat16)

    def driver(x, topk_weights, topk_ids):
        assert x.dtype is torch.bfloat16 and topk_weights.dtype is torch.float32
        assert topk_ids.dtype is torch.int32
        return x * 0.5

    experts = PreparedTrellisExperts(source, ids, weights, output, driver=driver, name="test")
    assert experts.functional(session=None) is experts
    result = experts(source, ids, weights, router_logits=None, selected_logits=None, image_mask=None)
    assert result.shape == (2, 16) and torch.equal(result, source * 0.5)


def test_tables_expose_the_device_token_map():
    # Regression: the TrellisMX rebase dropped main's cached property and the
    # loss only appeared when the graph catalog asked for the token map.
    from functools import cached_property
    assert isinstance(Tables.__dict__.get("device_token_map"), cached_property)
