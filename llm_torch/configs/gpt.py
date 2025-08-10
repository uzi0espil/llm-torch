import torch

from llm_torch import configs
from llm_torch.components import callbacks


GPT2_CONFIG_124 = configs.LLMConfig(
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
       n_heads=12,
       n_layers=12,
       hidden_dim=3072,  # emb_dim * 4
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