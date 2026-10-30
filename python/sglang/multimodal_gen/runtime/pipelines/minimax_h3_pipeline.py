# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    MiniMaxH3SamplingParams,
    PikaH3FlexSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3AudioEncodingStage,
    MiniMaxH3DecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3TextEncodingStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_TASK_PARTITIONS,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MiniMaxH3Pipeline"
    default_model_subfolder = "FL2VA"
    is_video_pipeline = True
    pipeline_config_cls = MiniMaxH3PipelineConfig
    sampling_params_cls = MiniMaxH3SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "audio_vae",
        # scheduler intentionally absent: model_index carries scheduler=null;
        # per-modality sigma schedules are generated in TimestepPreparation
        # from the task profile, and the loop scheduler math lives in
        # minimax_h3/denoise_loop.py (stages accept scheduler=None).
        "transformer",
    ]

    def __init__(self, *args, **kwargs):
        # TODO: Enable this check on ROCm after adding ffmpeg to the AMD Docker
        # image and CI dependency installer.
        if not current_platform.is_rocm():
            missing_media_tools = [
                executable
                for executable in ("ffmpeg", "ffprobe")
                if shutil.which(executable) is None
            ]
            if missing_media_tools:
                raise RuntimeError(
                    "MiniMax H3 requires ffmpeg and ffprobe for media processing "
                    "and validated output delivery; missing executables: "
                    f"{', '.join(missing_media_tools)}. Install the ffmpeg system "
                    "package before starting SGLang."
                )
        super().__init__(*args, **kwargs)

    @staticmethod
    def model_subfolder_for_variant(variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MiniMax H3 model variant must be a non-empty string")
        normalized = variant.strip().lower()
        subfolders = {
            "fl2va": "FL2VA",
            "ref2va": "Ref2VA",
            "hybrid": "Ref2VA",
        }
        try:
            return subfolders[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MiniMax H3 model variant {variant!r}; "
                f"supported: {sorted(subfolders)!r}"
            ) from exc

    def _load_config(self):
        model_variant = self.server_args.model_variant
        if model_variant is not None:
            semantic_subfolder = self.model_subfolder_for_variant(model_variant)
            if model_variant.strip().lower() == "hybrid" and not (
                self.server_args.component_weights_paths.get("transformer")
                or self.server_args.transformer_weights_path
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant hybrid requires explicit merged "
                    "weights via --component-weights-paths.transformer"
                )
            explicit_subfolder = self.server_args.model_subfolder
            if (
                explicit_subfolder is not None
                and explicit_subfolder.strip().lower() != semantic_subfolder.lower()
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant and --model-subfolder select "
                    f"different weight partitions: variant={model_variant!r} maps to "
                    f"{semantic_subfolder!r}, model_subfolder="
                    f"{explicit_subfolder!r}"
                )
            self.server_args.model_subfolder = semantic_subfolder
        model_index = super()._load_config()
        self.release_metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
        if (
            model_variant is not None
            and self.release_metadata.partition != semantic_subfolder.lower()
        ):
            raise ValueError(
                "MiniMax H3 loaded checkpoint partition does not match "
                f"--model-variant {model_variant!r}"
            )
        if model_variant is not None and model_variant.strip().lower() == "hybrid":
            # merged checkpoints share the native graph across all three tasks
            # keep the base partition contract strict unless explicitly selected
            self.release_metadata = replace(
                self.release_metadata,
                partition="hybrid",
                tasks=tuple(MINIMAX_H3_TASK_PARTITIONS),
            )
        return model_index

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "MiniMaxH3Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.pdd_heads = self._resolve_pdd_heads()
        transformer = self.get_module("transformer")
        if (
            self.pdd_heads
            and transformer is not None
            and transformer.final_layer._pdd_heads is None
        ):
            transformer.final_layer.load_pdd_fused_heads(self.pdd_heads)

    def _resolve_pdd_heads(self) -> str | None:
        """The env override, else a PDD release's own fused heads."""
        if envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS:
            return envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS
        release_metadata = getattr(self, "release_metadata", None)
        if release_metadata is None or release_metadata.pdd_fused_heads is None:
            return None
        return os.path.join(self.model_path, release_metadata.pdd_fused_heads)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Per-model sigma override from model_index.json; contract tests
        # construct the pipeline without model_path, hence the guard.
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        self.add_stage(
            MiniMaxH3TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            )
        )
        self.add_stage(
            MiniMaxH3VisualEncodingStage(
                video_vae=self.get_module("video_vae"),
                vae_arch_config=server_args.pipeline_config.vae_config.arch_config,
            )
        )
        self.add_stage(
            MiniMaxH3AudioEncodingStage(
                audio_vae=self.get_module("audio_vae"),
                vae_arch_config=server_args.pipeline_config.audio_vae_config.arch_config,
            )
        )
        self.add_stage(MiniMaxH3LatentPreparationStage())
        self.add_stage(
            MiniMaxH3TimestepPreparationStage(
                sigma_shift_scales=sigma_shift_scales,
                dmd_denoising_steps=(
                    release_metadata.dmd_denoising_steps
                    if release_metadata is not None
                    else None
                ),
                pdd_heads=getattr(self, "pdd_heads", None),
            )
        )
        self.add_stage(
            MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
                sampler=release_metadata.sampler
                if release_metadata is not None
                else "euler",
            )
        )
        self.add_stage(
            MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )

    def create_comfyui_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
            MiniMaxH3ComfyUIStepStage,
        )

        self.add_stage(
            MiniMaxH3ComfyUIStepStage(transformer=self.get_module("transformer"))
        )


class PikaH3FlexPipeline(MiniMaxH3Pipeline):
    pipeline_name = "PikaH3FlexPipeline"
    pipeline_config_cls = PikaH3FlexPipelineConfig
    sampling_params_cls = PikaH3FlexSamplingParams
    model_revision = "0ddf55ad7067ca1e76812253bd7b4600df87f759"
    base_repo = "MiniMaxAI/MiniMax-H3"
    base_revision = "42ed227ee7df40d41602854ae760620d6eb651fe"

    def load_lora_adapter(
        self,
        lora_path: str,
        lora_nickname: str,
        rank: int,
        weight_name: str | None = None,
        lora_alpha: int | None = None,
    ) -> None:
        super().load_lora_adapter(
            lora_path, lora_nickname, rank, weight_name, lora_alpha
        )
        # Release files store FP32 masters; inference uses BF16 adapter factors.
        self.lora_adapters[lora_nickname] = {
            name: weight.to(torch.bfloat16)
            for name, weight in self.lora_adapters[lora_nickname].items()
        }

    def _load_config(self):
        # The released adapters require FP32 matmuls at FP32 module boundaries.
        torch.set_float32_matmul_precision("highest")
        variant = self.server_args.model_variant or "fl2va"
        if self.server_args.revision not in (None, self.model_revision):
            raise ValueError(
                f"Pika H3 Flex requires model revision {self.model_revision}"
            )
        adapter_root = Path(self.model_path)
        if not adapter_root.is_dir():
            adapter_root = Path(
                snapshot_download(
                    self.model_path,
                    revision=self.model_revision,
                    allow_patterns=[f"{variant}/*", "LICENSE", "NOTICE", "licenses/*"],
                )
            )
        with (adapter_root / variant / "manifest.json").open() as file:
            manifest = json.load(file)
        expected_policy = (
            "fp32_conditioning_v1" if variant == "ref2va" else "bf16_conditioning_v1"
        )
        expected_sampling = {
            "knots": [1000, 750, 500, 250, 100, 40, 0],
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "method": "renoise",
        }
        expected = {
            "format_version": 1,
            "rank": 128,
            "alpha": 128,
            "task": variant,
            "trunk": "transformer_ref" if variant == "ref2va" else "transformer",
            "numerics_policy": expected_policy,
            "sampling": expected_sampling,
            "base": {"repo_id": self.base_repo, "revision": self.base_revision},
            "target_modules": [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"Pika H3 Flex adapter manifest has an unsupported {name}"
                )
        adapter_path = adapter_root / variant / "adapter.safetensors"
        checksum = hashlib.sha256()
        with adapter_path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != manifest.get("sha256"):
            raise ValueError("Pika H3 Flex adapter checksum mismatch")
        subfolder = self.model_subfolder_for_variant(variant)
        self.model_path = snapshot_download(
            self.base_repo,
            revision=self.base_revision,
            allow_patterns=[f"{subfolder}/**", "LICENSE", "NOTICE", "licenses/*"],
        )
        self.server_args.model_variant = variant
        self.server_args.lora_path = str(adapter_path)
        self.server_args.lora_alpha = manifest["alpha"]
        model_index = super()._load_config()
        model_index["_minimax_h3"].update(
            dmd_denoising_steps=manifest["sampling"]["knots"][:-1],
            sigma_shift_scales={"video": 12.0, "audio": 3.0},
            sampler=manifest["sampling"]["method"],
        )
        self.release_metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
        return model_index


class FastH3Pipeline(MiniMaxH3Pipeline):
    """FastH3: 8-step DMD2-distilled MiniMax-H3 (V2, t2va only), and the
    FastH3 Omni Ref2VA PDD student.

    The flat single-partition repo is materialized into the base-H3 layout by
    its registered model overlay (see runtime/utils/model_overlay.py), so every stage, loader,
    and admission path below is exactly the MiniMax-H3 one. There is no
    FL2VA/Ref2VA partition layout to default into.
    """

    pipeline_name = "FastH3Pipeline"
    default_model_subfolder = None


EntryClass = [MiniMaxH3Pipeline, FastH3Pipeline, PikaH3FlexPipeline]
