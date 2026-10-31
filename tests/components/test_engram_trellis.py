# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import torch

from ignition.components.engram_trellis import (
    reference_decode_rows, reference_decode_rows_unpack, state_lut_bytes,
    tensor_core_permutation,
)


def test_reference_decoders_agree_on_synthetic_rows():
    generator = torch.Generator().manual_seed(17)
    trellis = torch.randint(0, 256, (6, 128), generator=generator, dtype=torch.uint8)
    scales = torch.full((6, 8), 127, dtype=torch.uint8)  # UE8M0 exponent 0
    design = reference_decode_rows(trellis, scales)
    packer = reference_decode_rows_unpack(trellis, scales)
    assert design.shape == (6, 256)
    assert torch.equal(design, packer)


def test_reference_decoders_reject_wrong_row_width():
    with pytest.raises(ValueError, match="uint8 .*128"):
        reference_decode_rows(torch.zeros((1, 127), dtype=torch.uint8),
                              torch.zeros((1, 8), dtype=torch.uint8))


def test_state_lut_and_permutation_are_the_pinned_owner_tables():
    # Independently reproduced from the pack's author reference codec
    # (reference/author/codec.py, state_lut_e4m3_mcg_alpha2). If this digest
    # changes, every K4 Engram lookup silently decodes a different alphabet.
    lut = state_lut_bytes().numpy().tobytes()
    assert hashlib.sha256(lut).hexdigest() == (
        "f81eb38a599601d856c28a8c6c6e334d57f547827124c452655c0e0ada6fe1ec")
    assert len(set(lut)) == 161
    perm = tensor_core_permutation().to(torch.int32).numpy().tobytes()
    assert hashlib.sha256(perm).hexdigest() == (
        "6c02541e911279020153671b707c25d80b2267b116074acede23400121086851")
