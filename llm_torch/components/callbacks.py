import logging
import math
import os
import numpy as np


logger = logging.getLogger(__name__)


class Callback:

    def __init__(self):
        self.trainer = None

    def setup(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, total_steps):
        pass

    def on_train_end(self):
        pass

    def on_train_batch_begin(self, batch, stats: dict):
        pass

    def on_train_batch_end(self, batch, stats: dict):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics: dict):
        pass

    def state_dict(self):
        return {}

    @classmethod
    def load_state_dict(cls, state: dict, **kwargs):
        init_args = {}
        other_attrs = {}
        for key, value in state.items():
            if key.startswith('_') or key.endswith('_'):
                other_attrs[key] = value
            else:
                init_args[key] = value

        init_args.update(kwargs)
        instance = cls(**init_args)

        for key, value in other_attrs.items():
            setattr(instance, key, value)

        return instance


class LRCosineAnnealing(Callback):

    def __init__(self,
                 initial_lr: float = 0.,
                 min_lr: float = 0.,
                 peak_lr: float = 2.5e-4,
                 warmup_steps: float | int = 0.2):
        super().__init__()
        if peak_lr <= initial_lr:
            raise ValueError("initial lr must be smaller than peak lr")
        if peak_lr <= min_lr:
            raise ValueError("minimum lr must be smaller than peak lr")
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        self.current_lr_ = initial_lr
        self.lr_increment_ = 0.
        self.lrs_ = []
        self._finished_warmup_phase = False
        self.total_steps_ = 0.

    def _validate_warmup_steps(self, total_steps=None):
        if isinstance(self.warmup_steps, float):
            if total_steps is None:
                raise ValueError("total_steps must not be None if warmup_steps is percentage.")
            if self.warmup_steps > 0.0 or self.warmup_steps <= 1.0:
                return self.warmup_steps // total_steps
            else:
                raise ValueError("warmup_steps must be a float between 0.0 and 1.0")
        elif isinstance(self.warmup_steps, int):
            return self.warmup_steps
        else:
            raise ValueError("warmup_steps must be a percentage (float) or number of steps (int).")

    def set_lr(self, lr):
        self.lrs_.append(lr)
        for param_group in self.trainer.optimizer.param_groups:
            param_group["lr"] = lr

    def on_train_begin(self, total_steps):
        self.total_steps_ = total_steps
        if not self._finished_warmup_phase:
            self.warmup_steps = self._validate_warmup_steps(total_steps)
            self.lr_increment_ = (self.peak_lr - self.initial_lr) / total_steps
        self.set_lr(self.initial_lr)

    def _cosine_annealing(self):
        progress = (self.trainer.global_step_ - self.warmup_steps) / (self.total_steps_ - self.warmup_steps)
        return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def on_train_batch_end(self, batch, states: dict):
        if self.trainer.global_step_ < self.warmup_steps:
            self.current_lr_ = self.initial_lr + self.trainer.global_step_ * self.lr_increment_
        else:
            if not self._finished_warmup_phase and self.warmup_steps > 0:
                logger.debug("Warmup phase finished, cosine annealing started.")
                self._finished_warmup_phase = True
            self.current_lr_ = self._cosine_annealing()
        self.set_lr(self.current_lr_)
        states.update({"current_lr": self.current_lr_})

    def state_dict(self):
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "current_lr_": self.current_lr_,
            "lrs_": self.lrs_,
            "_finished_warmup_phase": self._finished_warmup_phase,
            "lr_increment_": self.lr_increment_,
            "total_steps_": self.total_steps_,
        }


class GenerateSample(Callback):

    def __init__(self, start_context, max_new_tokens, tokenizer,
                 eos_id=None, pad_id=0, temperature=1.0, top_k=5, device='cpu'):
        super().__init__()
        self.start_context = start_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device = device

    def on_epoch_end(self, epoch, metrics):
        from llm_torch.engine.predictor import Predictor
        predictor = Predictor(self.trainer.model, self.tokenizer, eos_id=self.eos_id,
                              pad_id=self.pad_id, device=self.device)
        text = predictor.predict(self.start_context, self.max_new_tokens, self.temperature, self.top_k)
        print(f"Generated sample: {text}")

    def state_dict(self):
        # todo: a bit difficult to save currently due to serializing the tokenizer.
        logger.debug("GenerateSample cannot be stored, when loaded again, it will be lost.")
        return None


class ModelCheckpoint(Callback):
    """
    Callback to save the model checkpoint.
    Args:
        filepath (str): Path to save the model file. Can contain placeholders
            like `{epoch}` and `{step}` and any metric keys.
        monitor (str): The metric to monitor for saving the best model.
        mode (str): One of 'min' or 'max'. In 'min' mode, saving is triggered
            when the monitored metric decreases. In 'max' mode, it's triggered
            on an increase.
        save_best_only (bool): If True, only the best model is saved.
        frequency (str or int): If 'epoch', saved every epoch. If an integer,
            saved every `frequency` steps.
        verbose (int): Verbosity mode.
    """
    def __init__(self,
                 filepath: str = "model_{epoch}.pth",
                 monitor='val_loss',
                 mode='min',
                 save_best_only=False,
                 frequency='epoch',
                 verbose=0):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.frequency = frequency
        self.verbose = verbose

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', but got {mode}")

        if self.mode == 'min':
            self._best_metric = np.inf
        else:
            self._best_metric = -np.inf
        
        self._last_saved_path = None

    def _save_checkpoint(self, filepath):
        if self.trainer is None:
            logger.warning("Trainer is not set. Cannot save checkpoint.")
            return
        
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # Assuming trainer has a `save` method that saves all necessary states
        self.trainer.model.overall_state_dict(filepath)
        if self.verbose > 0:
            logger.info(f"Checkpoint saved to {filepath}")
        self._last_saved_path = filepath

    def _format_filepath(self, **kwargs):
        try:
            return self.filepath.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Filepath formatting failed. Missing key: {e}. Using raw filepath.")
            return self.filepath

    def on_epoch_end(self, epoch, metrics):
        if self.trainer is None:
            return
        
        if self.save_best_only:
            current_metric = metrics.get(self.monitor)
            if current_metric is None:
                if self.trainer.global_step_ == 0: # Don't warn on first step
                    logger.info(f"Metric '{self.monitor}' not found, skipping.")
                else:
                    logger.warning(f"Metric '{self.monitor}' not found in metrics. Skipping checkpoint.")
                return

            if (self.mode == 'min' and current_metric < self._best_metric) or \
               (self.mode == 'max' and current_metric > self._best_metric):
                
                if self.verbose > 0:
                    logger.info(f"Metric {self.monitor} improved from {self._best_metric:.4f} to {current_metric:.4f}. Saving model.")
                
                self._best_metric = current_metric
                filepath = self._format_filepath(epoch=epoch, **metrics)
                self._save_checkpoint(filepath)
        
        elif self.frequency == 'epoch':
            filepath = self._format_filepath(epoch=epoch, **metrics)
            self._save_checkpoint(filepath)

    def on_train_batch_end(self, batch, stats: dict):
        if self.save_best_only or self.frequency == 'epoch' or not isinstance(self.frequency, int) or self.frequency <= 0:
            return

        if self.trainer and self.trainer.global_step_ > 0 and self.trainer.global_step_ % self.frequency == 0:
            filepath = self._format_filepath(step=self.trainer.global_step_, **stats)
            self._save_checkpoint(filepath)

    def state_dict(self):
        return {
            'filepath': self.filepath,
            'monitor': self.monitor,
            'mode': self.mode,
            'save_best_only': self.save_best_only,
            'frequency': self.frequency,
            'verbose': self.verbose,
            '_best_metric': self._best_metric,
            '_last_saved_path': self._last_saved_path,
        }