"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.datasets.semantic_dataset import SemanticDataset

from my_semantic.my_semantic_dataparser import MySemanticDataParserConfig, MySemantic
from my_semantic.my_semantic_model import MySemanticNerfWModelConfig
# nerfstudio/configs/method_configs.py
my_semantic = MethodSpecification(
    config=TrainerConfig(
        method_name="my-semantic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[SemanticDataset],
                dataparser=MySemanticDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=8192,
            ),
            model=MySemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)
