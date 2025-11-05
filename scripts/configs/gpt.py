import torch

from llm_torch import configs
from llm_torch.components import callbacks


GPT2_CONFIG_124M = configs.LLMConfig(
    vocab_size=50257,
    context_length=256,
    dataset_config=configs.DatasetConfig(
        batch_size=32,  # it originally trained on 512
        shuffle=True,
        max_length=256,
        stride=1,
    ),
    model_config=configs.ModelConfig(
        emb_dim=768,
        n_layers=12,
        ff_block_config=configs.FFBlockConfig(
            hidden_dim=3072,
        ),
        normalizer_config=configs.LayerNormConfig(),
        attention_config=configs.MultiHeadAttentionConfig(
            n_heads=12,
        ),
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


GPT_OSS_CONFIG_20B = configs.LLMConfig(
    vocab_size=50257,  # 201088
    context_length=256,
    dataset_config=configs.DatasetConfig(
        batch_size=32,  # it originally trained on 512
        shuffle=True,
        max_length=256,
        stride=1,
    ),
    model_config=configs.ModelConfig(
        emb_dim=2880,
        dtype=torch.bfloat16,
        n_layers=36,
        ff_block_config=configs.MoEConfig(
            hidden_dim=2880,
            n_experts=128,
            n_experts_per_token=4,
            ff_block=configs.SwiGLUBlockConfig(
                hidden_dim=2880,
                activation=configs.SiLUConfig(alpha=1.702),
                limit=7.0
            ),
        ),
        normalizer_config=configs.LayerNormConfig(),
        attention_config=configs.YarnSWAConfig(
            n_heads=64,
            head_dim=64,
            n_kv_group=8,
            window_size=128,
            factor=32,
            low_freq=1.0,
            high_freq=32.0,
            theta_base=150000.0,
            original_max_pos_embeddings=4096
        ),
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
