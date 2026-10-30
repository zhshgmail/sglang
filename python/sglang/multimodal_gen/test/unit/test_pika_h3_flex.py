# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex registration and released sampler contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import PikaH3FlexSamplingParams
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
    has_registered_diffusion_model_path,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    prequeue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)


def test_pika_registry_reuses_h3_without_shadowing_the_base():
    info = get_model_info("Pika-Labs/Pika-H3-Flex")
    assert info.pipeline_config_cls is PikaH3FlexPipelineConfig
    assert info.sampling_param_cls is PikaH3FlexSamplingParams
    assert (
        get_non_diffusers_pipeline_name("Pika-Labs/Pika-H3-Flex")
        == "PikaH3FlexPipeline"
    )
    assert (
        get_model_info("MiniMaxAI/MiniMax-H3").pipeline_config_cls
        is MiniMaxH3PipelineConfig
    )


@pytest.mark.parametrize(
    "path",
    [
        "other-org/Pika-H3-Flex",
        "Pika-Labs/Pika-H3-Flex-extra",
        "/cache/models--pika-labs--pika-h3-flex-extra/snapshots/abc",
    ],
)
def test_pika_registry_requires_the_exact_repo_id(path):
    assert get_non_diffusers_pipeline_name(path) is None
    assert not has_registered_diffusion_model_path(path)


def test_pika_registry_accepts_the_pinned_cache_directory():
    path = "/cache/models--Pika-Labs--Pika-H3-Flex/snapshots/abc"
    assert get_non_diffusers_pipeline_name(path) == "PikaH3FlexPipeline"
    assert get_model_info(path).pipeline_config_cls is PikaH3FlexPipelineConfig


@pytest.mark.parametrize("steps", [1, 5, 7, 50])
def test_pika_rejects_other_step_counts(steps):
    with pytest.raises(ValueError, match="exactly six"):
        PikaH3FlexSamplingParams(prompt="a waterfall", num_inference_steps=steps)


def test_sampler_is_selected_by_metadata_and_unknown_values_fail():
    index = {
        "_minimax_h3": {
            "schema_version": 1,
            "partition": "fl2va",
            "tasks": ["t2va", "fl2va"],
            "sigma_shift_scales": {"video": 12, "audio": 3},
            "dmd_denoising_steps": [1000, 750, 500, 250, 100, 40],
        }
    }
    assert MiniMaxH3ReleaseMetadata.from_model_index(index).sampler == "euler"
    index["_minimax_h3"]["sampler"] = "renoise"
    assert MiniMaxH3ReleaseMetadata.from_model_index(index).sampler == "renoise"
    index["_minimax_h3"]["sampler"] = "unknown"
    with pytest.raises(ValueError, match="sampler"):
        MiniMaxH3ReleaseMetadata.from_model_index(index)


@pytest.mark.parametrize(
    "block", ["transformer_blocks.49", "token_refiner.refiner_blocks.1"]
)
def test_pika_peft_format_preserves_tensor_pairs(block):
    weights = {
        f"{block}.attn.to_q.lora_A.default.weight": torch.randn(128, 8),
        f"{block}.attn.to_q.lora_B.default.weight": torch.randn(16, 128),
    }
    assert detect_lora_format_from_state_dict(weights) is LoRAFormat.STANDARD
    normalized = normalize_lora_state_dict(weights)
    assert set(normalized) == {key.replace(".default.", ".") for key in weights}
    for name, weight in weights.items():
        assert normalized[name.replace(".default.", ".")] is weight


def _pika_request(task="t2va", conditions=None, **target):
    return PikaH3FlexSamplingParams(
        prompt="a waterfall",
        task=task,
        conditions=conditions or [],
        seed=42,
        target={
            "short_edge": 480,
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            **target,
        },
    )


def _plan(params):
    return minimax_h3_resolve_plan(
        params.build_request_extra()["minimax_h3_canonical_request"]
    )


@pytest.mark.parametrize(
    "tier,aspect,expected",
    [
        (480, "21:9", (960, 416)),
        (480, "16:9", (832, 480)),
        (480, "3:2", (704, 480)),
        (480, "4:3", (640, 480)),
        (480, "5:4", (608, 480)),
        (480, "1:1", (480, 480)),
        (480, "4:5", (480, 608)),
        (480, "3:4", (480, 640)),
        (480, "2:3", (480, 704)),
        (480, "9:16", (480, 832)),
        (480, "9:21", (416, 960)),
        (768, "21:9", (1536, 672)),
        (768, "16:9", (1344, 768)),
        (768, "3:2", (1152, 768)),
        (768, "4:3", (1024, 768)),
        (768, "5:4", (960, 768)),
        (768, "1:1", (768, 768)),
        (768, "4:5", (768, 960)),
        (768, "3:4", (768, 1024)),
        (768, "2:3", (768, 1152)),
        (768, "9:16", (768, 1344)),
        (768, "9:21", (672, 1536)),
    ],
)
def test_released_buckets_and_aligned_time(tier, aspect, expected):
    plan = _plan(_pika_request(short_edge=tier, aspect_ratio=aspect))
    assert (plan.shape["width"], plan.shape["height"]) == expected
    assert plan.shape["frame_count"] == 124
    assert plan.shape["fps"] == 24
    assert _plan(_pika_request(duration_seconds=15)).shape["frame_count"] == 362


@pytest.mark.parametrize(
    "task,indices", [("i2va", [0]), ("l2va", [-1]), ("fl2va", [0, -1])]
)
def test_offline_and_http_aliases_share_keyframe_plan(task, indices):
    conditions = [
        dict(type="image", role="keyframe", uri=f"frame{i}.png", frame_index=i)
        for i in indices
    ]
    params = _pika_request(task, conditions)
    request = VideoGenerationsRequest(
        prompt=params.prompt, task=task, conditions=conditions, target=params.target
    )
    kwargs = PikaH3FlexSamplingParams.lower_video_request_kwargs(
        request, {"prompt": params.prompt}
    )
    http_plan = _plan(PikaH3FlexSamplingParams(**kwargs))
    plan = _plan(params)
    assert http_plan.shape == plan.shape
    assert plan.task == "fl2va"
    assert plan.condition_mask["semantic_frame_indices"] == indices
    assert plan.shape["keyframe_fit"] == "source_crop"


@pytest.mark.parametrize(
    "target",
    [
        {"short_edge": 512},
        {"short_edge": 1088},
        {"aspect_ratio": "auto"},
        {"aspect_ratio": "2:1"},
    ],
)
def test_pika_rejects_unreleased_geometry(target):
    with pytest.raises(ValueError, match="Pika H3 Flex target"):
        _plan(_pika_request(**target))


@pytest.mark.parametrize("kind,count", [("image", 10), ("video", 4), ("audio", 4)])
def test_reference_count_limits(kind, count):
    refs = [dict(type=kind, role="reference", uri=f"ref{i}") for i in range(count)]
    with pytest.raises(ValueError, match="reference limits"):
        _plan(_pika_request("ref2va", refs))


def test_reference_requires_visual_input():
    with pytest.raises(ValueError, match="at least one image or video"):
        _plan(
            _pika_request(
                "ref2va", [dict(type="audio", role="reference", uri="audio.wav")]
            )
        )


@pytest.mark.parametrize(
    "durations,accepted",
    [([2, 13], True), ([1.99], False), ([15.01], False), ([8, 8], False)],
)
def test_reference_duration_limits_before_queue(monkeypatch, durations, accepted):
    refs = [
        dict(type="video", role="reference", uri=f"ref{i}.mp4")
        for i in range(len(durations))
    ]
    params = _pika_request("ref2va", refs)
    batch = SimpleNamespace(sampling_params=params, extra=params.build_request_extra())

    def probe(batch, uri, *, condition_type, condition_index):
        return {
            "display_width": 1920,
            "display_height": 1080,
            "video_duration_seconds": durations[condition_index],
        }

    monkeypatch.setattr(prequeue, "minimax_h3_probe_material", probe)
    if accepted:
        plan = prequeue.minimax_h3_prepare_for_queue(batch)
        assert (plan.shape["width"], plan.shape["height"]) == (832, 480)
        shapes = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY]
        assert (shapes[0]["width"], shapes[0]["height"]) == (832, 480)
    else:
        with pytest.raises(ValueError, match="Pika H3 Flex"):
            prequeue.minimax_h3_prepare_for_queue(batch)


@pytest.mark.parametrize("tier,expected", [(480, (1536, 1024)), (768, (3072, 2048))])
def test_reference_image_uses_output_tier(monkeypatch, tier, expected):
    params = _pika_request(
        "ref2va", [dict(type="image", role="reference", uri="ref.png")], short_edge=tier
    )
    batch = SimpleNamespace(sampling_params=params, extra=params.build_request_extra())
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *a, **kw: {"display_width": 600, "display_height": 400},
    )
    prequeue.minimax_h3_prepare_for_queue(batch)
    shape = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY][0]
    assert (shape["width"], shape["height"]) == expected


@pytest.mark.parametrize(
    "path,model_id",
    [
        ("/cache/models--Pika-Labs--Pika-H3-Flex/snapshots/revision", None),
        ("/models/local-adapter-pack", "Pika-H3-Flex"),
    ],
)
def test_local_pika_pack_resolves_to_the_native_variant(path, model_id):
    assert (
        get_model_info(path, model_id=model_id).pipeline_config_cls
        is PikaH3FlexPipelineConfig
    )


@pytest.mark.parametrize("variant", ["fl2va", "ref2va"])
def test_release_manifest_pins_the_base_and_matching_adapter(
    tmp_path, monkeypatch, variant
):
    import hashlib
    import json

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline as pipelines

    pipeline = pipelines.PikaH3FlexPipeline.__new__(pipelines.PikaH3FlexPipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.server_args = SimpleNamespace(model_variant=variant, revision=None)
    partition = tmp_path / variant
    partition.mkdir()
    adapter = partition / "adapter.safetensors"
    adapter.write_bytes(b"adapter-checksum-fixture")
    manifest = {
        "format_version": 1,
        "rank": 128,
        "alpha": 128,
        "task": variant,
        "trunk": "transformer_ref" if variant == "ref2va" else "transformer",
        "numerics_policy": "fp32_conditioning_v1"
        if variant == "ref2va"
        else "bf16_conditioning_v1",
        "sampling": {
            "knots": [1000, 750, 500, 250, 100, 40, 0],
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "method": "renoise",
        },
        "base": {"repo_id": pipeline.base_repo, "revision": pipeline.base_revision},
        "target_modules": [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ],
        "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    }
    manifest_path = partition / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    downloads = []

    def download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))
        return "/pinned-base"

    monkeypatch.setattr(pipelines, "snapshot_download", download)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _: None)
    monkeypatch.setattr(
        pipelines.MiniMaxH3Pipeline,
        "_load_config",
        lambda _: {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": variant,
                "tasks": ["ref2va"] if variant == "ref2va" else ["t2va", "fl2va"],
            }
        },
    )
    pipeline._load_config()
    assert pipeline.release_metadata.sampler == "renoise"
    assert pipeline.release_metadata.dmd_denoising_steps == (
        1000,
        750,
        500,
        250,
        100,
        40,
    )
    assert pipeline.server_args.lora_path == str(adapter)
    assert pipeline.server_args.lora_alpha == 128
    assert downloads[0][0] == "MiniMaxAI/MiniMax-H3"
    assert downloads[0][1]["revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
    assert downloads[0][1]["allow_patterns"][0] == (
        "Ref2VA/**" if variant == "ref2va" else "FL2VA/**"
    )

    for field, invalid in (
        ("rank", 64),
        ("base", {"repo_id": pipeline.base_repo, "revision": "main"}),
        ("sampling", {**manifest["sampling"], "method": "euler"}),
        ("sha256", "corrupt"),
    ):
        pipeline.model_path = str(tmp_path)
        manifest_path.write_text(json.dumps({**manifest, field: invalid}))
        with pytest.raises(ValueError, match="manifest|checksum"):
            pipeline._load_config()
    assert len(downloads) == 1


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_condition_noise_preserves_clean_cache_and_release_precision(task):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        _apply_condition_noise_aug,
    )

    clean = torch.tensor([[1.003, -0.12345, 7.997]], dtype=torch.float32)
    original = clean.clone()
    audio = torch.tensor([[0.123456]], dtype=torch.float32)
    sampling = PikaH3FlexSamplingParams(prompt="a lake", seed=42)
    contexts = [
        SimpleNamespace(
            cond_rows=clean,
            audio_cond_rows=audio,
            plan=SimpleNamespace(task=task, seed=42),
        )
        for _ in range(2)
    ]
    for context in contexts:
        _apply_condition_noise_aug(
            context, sampling=sampling, imgvid_noise_aug=0.999, audio_noise_aug=1.0
        )
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42 + 7919))
    rounded = clean.bfloat16().float() if task == "fl2va" else clean
    torch.testing.assert_close(contexts[0].cond_rows, 0.999 * rounded + 0.001 * noise)
    assert torch.equal(contexts[0].cond_rows, contexts[1].cond_rows)
    assert torch.equal(clean, original)
    assert contexts[0].audio_cond_rows is audio


def test_server_cache_default_cannot_override_the_six_step_contract(monkeypatch):
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    monkeypatch.setattr(
        MiniMaxH3PipelineConfig, "validate_server_args", lambda *_: None
    )
    args = SimpleNamespace(
        model_variant="fl2va",
        lora_path=None,
        lora_alpha=None,
        lora_scale=1.0,
        is_arg_explicitly_set=lambda _: False,
    )
    with pytest.raises(ValueError, match="disable SGLANG_CACHE_DIT_ENABLED"):
        PikaH3FlexPipelineConfig().validate_server_args(args)
