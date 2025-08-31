import torch

from llm_torch import configs
from llm_torch.components import callbacks
from llm_torch.configs.configs import RoPEConfig

LLAMA2_CONFIG_7B = configs.LLMConfig(
    vocab_size=50257,  # 32000,
    context_length=256,
    dataset_config=configs.DatasetConfig(
       batch_size=32,  # it originally trained on 512
       shuffle=True,
       max_length=256,
       stride=1,
    ),
    model_config=configs.ModelConfig(
       emb_dim=768, #4096,
       hidden_dim=11008,
       n_heads=2, #32,
       n_layers=2, #32,
       drop_rate=None,
       qkv_bias=False,
       dtype=torch.bfloat16,
       kv_window_size=256,
    ),
    train_config=configs.TrainerConfig(
       epochs=3,
       eval_freq=5,
       eval_iter=5,
       max_grad_norm=1.0,
       optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs = [
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


LLAMA3_CONFIG_8B = configs.LLMConfig(
    vocab_size=50257,  # 128_256,
    context_length=256,  # 4096
    dataset_config=configs.DatasetConfig(
       batch_size=32,  # it originally trained on 512
       shuffle=True,
       max_length=256,
       stride=1,
    ),
    model_config=configs.ModelConfig(
       emb_dim=1024, #4096,
       hidden_dim=2816,  #14_336
       n_heads=16, #32,
       n_kv_group=4, #8
       n_layers=12, #32,
       drop_rate=None,
       qkv_bias=False,
       dtype=torch.bfloat16,
       kv_window_size=None,
       rope_scaling=RoPEConfig(theta_base=500_000.0)
    ),
    train_config=configs.TrainerConfig(
       epochs=3,
       eval_freq=5,
       eval_iter=5,
       max_grad_norm=1.0,
       optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs = [
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


LLAMA31_CONFIG_8B = configs.LLMConfig(
    vocab_size=50257,  # 128_256,
    context_length=256,  # 131_072
    dataset_config=configs.DatasetConfig(
       batch_size=32,  # it originally trained on 512
       shuffle=True,
       max_length=256,
       stride=1,
    ),
    model_config=configs.ModelConfig(
       emb_dim=1024, #4096,
       hidden_dim=2816,  #14_336
       n_heads=16, #32,
       n_kv_group=4, #8
       n_layers=12, #32,
       drop_rate=None,
       qkv_bias=False,
       dtype=torch.bfloat16,
       kv_window_size=None,
       rope_scaling=configs.YarnConfig(
           theta_base=500_000.0,
           factor=8.0,
           low_freq=1.0,
           high_freq=4.0,
           original_max_pos_embeddings=None #8192,  # originally llama3.1 fine-tuned and not trained from scratch.
       )
    ),
    train_config=configs.TrainerConfig(
       epochs=3,
       eval_freq=5,
       eval_iter=5,
       max_grad_norm=1.0,
       optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs = [
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


LLAMA32_CONFIG_1B = configs.LLMConfig(
    vocab_size=50257,  # 128_256,
    context_length=256,  # 131_072
    dataset_config=configs.DatasetConfig(
       batch_size=32,  # it originally trained on 512
       shuffle=True,
       max_length=256,
       stride=1,
    ),
    model_config=configs.ModelConfig(
       emb_dim=2048,
       hidden_dim=8192,
       n_heads=32,
       n_kv_group=8,
       n_layers=16,
       drop_rate=None,
       qkv_bias=False,
       dtype=torch.bfloat16,
       kv_window_size=None,
       rope_scaling=configs.YarnConfig(
           theta_base=500_000.0,
           factor=32.0,
           low_freq=1.0,
           high_freq=4.0,
           original_max_pos_embeddings=None #8192,  # originally llama3.1 fine-tuned and not trained from scratch.
       )
    ),
    train_config=configs.TrainerConfig(
       epochs=3,
       eval_freq=5,
       eval_iter=5,
       max_grad_norm=1.0,
       optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs = [
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


LLAMA32_CONFIG_3B = configs.LLMConfig(
    vocab_size=50257,  # 128_256,
    context_length=256,  # 131_072
    dataset_config=configs.DatasetConfig(
       batch_size=32,  # it originally trained on 512
       shuffle=True,
       max_length=256,
       stride=1,
    ),
    model_config=configs.ModelConfig(
       emb_dim=3072,
       hidden_dim=8192,
       n_heads=24,
       n_kv_group=8,
       n_layers=28,
       drop_rate=None,
       qkv_bias=False,
       dtype=torch.bfloat16,
       kv_window_size=None,
       rope_scaling=configs.YarnConfig(
           theta_base=500_000.0,
           factor=32.0,
           low_freq=1.0,
           high_freq=4.0,
           original_max_pos_embeddings=None #8192,  # originally llama3.1 fine-tuned and not trained from scratch.
       )
    ),
    train_config=configs.TrainerConfig(
       epochs=3,
       eval_freq=5,
       eval_iter=5,
       max_grad_norm=1.0,
       optimizer=configs.configs.OptimizerConfig(class_name=torch.optim.AdamW)
    ),
    callback_configs = [
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
