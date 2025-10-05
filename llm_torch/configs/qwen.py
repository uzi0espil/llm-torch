import torch

from llm_torch import configs
from llm_torch.components import callbacks, normalizer


QWEN3_CONFIG_30B = configs.LLMConfig(
    vocab_size=50257,  # 151_936,
    context_length=256,  # 262_144
    dataset_config=configs.DatasetConfig(
        batch_size=32,  # it originally trained on 512
        shuffle=True,
        max_length=256,
        stride=1,
    ),
    model_config=configs.ModelConfig(
        emb_dim=2048,
        ff_block_config=configs.MoEConfig(
            hidden_dim=768,
            n_experts=128,
            n_experts_per_token=8,
        ),
        normalizer_config=configs.RMSNormConfig(dtype=torch.float32),
        attention_config=configs.YarnGroupedAttentionConfig(
            n_heads=32,
            n_kv_group=8,
            qkv_bias=False,
            qk_norm=configs.RMSNormConfig(dtype=torch.float32),
            kv_window_size=None,
            theta_base=10_000_000.0,
            factor=32.0,
            low_freq=1.0,
            high_freq=4.0,
            original_max_pos_embeddings=None
        ),
        n_layers=2,  # 48,
        drop_rate=None,
        dtype=torch.bfloat16,
    ),
    train_config=configs.TrainerConfig(
        epochs=3,
        eval_freq=5,
        eval_iter=5,
        max_grad_norm=1.0,
        optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs=[
        configs.CallbackConfig(
            class_name=callbacks.LRCosineAnnealing,
            config=dict(peak_lr=2.5e-4, min_lr=0, initial_lr=0, warmup_steps=2000)
        ),
        configs.CallbackConfig(
            class_name=callbacks.ModelCheckpoint,
            config=dict(save_best_only=True),
        )
    ]
)
