from __future__ import annotations
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time
from typing import List, Optional
import importlib

from llm_torch.configs import TrainerConfig
from llm_torch.components import Callback
from llm_torch.utils import AverageMeter


logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, model, config: TrainerConfig, callbacks: Optional[List[Callback]] = None, device: str = 'cpu'):
        self.config = config
        self.model = model
        self.model.to(device)
        self.optimizer = config.optimizer.instantiate(model.parameters())
        self.device = device
        self.callbacks = callbacks or []

        # initialize callbacks
        [callback.setup(self) for callback in self.callbacks]

        self.global_step_ = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:

        train_losses, val_losses, track_tokens_seen = [], [], []
        train_perplexities, val_perplexities = [], []
        batch_train_losses, batch_val_losses = [], []
        batch_train_perplexities, batch_val_perplexities = [], []
        tokens_seen = 0
        n_batches = len(train_loader)
        total_steps = n_batches * self.config.epochs

        [callback.on_train_begin(total_steps=total_steps) for callback in self.callbacks]

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = AverageMeter()
            epoch_perplexity = AverageMeter()
            epoch_start = time.time()
            batch_time_accum = 0.0

            desc = f"Epoch {epoch + 1}/{self.config.epochs}"
            pbar = tqdm(train_loader, total=n_batches, desc=desc, leave=True)

            [callback.on_epoch_begin(epoch) for callback in self.callbacks]
            for i, (b_input, b_target) in enumerate(pbar):
                pbar_stats = dict()
                [callback.on_train_batch_begin((b_input, b_target), pbar_stats) for callback in self.callbacks]

                # run
                t0 = time.time()
                loss = self.step(b_input, b_target)
                dt = time.time() - t0
                batch_time_accum += dt

                tokens_seen += b_input.numel()

                # accumulate the training loss.
                loss_item = loss.item()
                with torch.no_grad():
                    perplexity_item = torch.exp(loss).item()
                batch_train_losses.append(loss_item)
                batch_train_perplexities.append(perplexity_item)
                epoch_loss.update(loss_item)
                epoch_perplexity.update(perplexity_item)

                pbar_stats = dict(loss=f"{epoch_loss.avg:.3f}",
                                  perplexity=f"{epoch_perplexity.avg:.3f}",
                                  step=self.global_step_,
                                  time=f"{batch_time_accum:.3f}s")

                [callback.on_train_batch_end((b_input, b_target), pbar_stats) for callback in self.callbacks]

                # validation after eval_freq.
                if self.global_step_ % self.config.eval_freq == 0:
                    val_loss, val_perplexity = self.evaluate(val_loader, batch_size=self.config.eval_iter)
                    batch_val_losses.append(val_loss)
                    batch_val_perplexities.append(val_perplexity)
                    pbar_stats["val_loss"] = f"{val_loss:.3f}"
                    pbar_stats["val_perplexity"] = f"{val_perplexity:.3f}"
                    track_tokens_seen.append(tokens_seen)
                    self.model.train()

                pbar.set_postfix(pbar_stats)
                self.global_step_ += 1

            pbar.close()

            epoch_time = time.time() - epoch_start

            # end of epoch
            # 1. run on whole training set, deactivate dropout.
            train_loss, train_perplexity = self.evaluate(train_loader, batch_size=None)
            train_losses.append(train_loss)
            train_perplexities.append(train_perplexity)
            # 2. validate on all validation set
            val_loss, val_perplexity = self.evaluate(val_loader, batch_size=None)
            val_losses.append(val_loss)
            val_perplexities.append(val_perplexity)

            [callback.on_epoch_end(epoch, metrics=dict(train_loss=train_loss,
                                                       val_loss=val_loss)) for callback in self.callbacks]

            print(
                f"End of Epoch {epoch + 1}: in {epoch_time:.2f}s, "
                f"Avg Train loss {epoch_loss.avg:.3f}, "
                f"Avg Train perplexity {epoch_perplexity.avg:.3f}, "
                f"Full Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                f"Full Train perplexity {train_perplexity:.3f}, Val perplexity {val_perplexity:.3f}."
            )

        return dict(loss=train_losses,
                    perplexity=train_perplexities,
                    batch_train_losses=batch_train_losses,
                    batch_train_perplexities=batch_train_perplexities,
                    val_loss=val_losses,
                    val_perplexity=val_perplexities,
                    batch_val_losses=batch_val_losses,
                    batch_val_perplexities=batch_val_perplexities,
                    track_tokens_seen=track_tokens_seen)

    def step(self, b_input, b_target):
        self.optimizer.zero_grad()  # reset gradients
        loss = self.compute_loss(b_input, b_target)
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        return loss

    def evaluate(self, val_loader: DataLoader, batch_size=None):
        self.model.eval()
        total_loss = 0.0
        total_perplexity = 0.0

        if len(val_loader) == 0:
            return float('nan'), float('nan')
        elif batch_size is None:
            batch_size = len(val_loader)
        else:
            batch_size = min(batch_size, len(val_loader))

        with torch.no_grad():
            for i, (b_input, b_target) in enumerate(val_loader):
                if i < batch_size:
                    loss = self.compute_loss(b_input, b_target)
                    total_loss += loss.item()
                    total_perplexity += torch.exp(loss).item()
                else:
                    break
        return total_loss / batch_size, total_perplexity / batch_size

    def compute_loss(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return self.config.loss(self.model(x), y)

    def save(self, path: str):
        state = {
            "model_state_dict": {
                "class_name": self.model.__class__.__name__,
                "module": self.model.__class__.__module__,
                "state": self.model.overall_state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step_": self.global_step_,
            "callback_states": [
                {
                    "class_name": cb.__class__.__name__,
                    "module": cb.__class__.__module__,
                    "state": cb.state_dict(),
                }
                for cb in self.callbacks if cb.state_dict() is not None
            ],
        }
        torch.save(state, path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = 'cpu', **kwargs):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = state["config"]

        # load the callbacks
        callbacks = []
        for cb_state in state.get("callback_states", []):
            module_path = cb_state["module"]
            class_name = cb_state["class_name"]
            state_dict = cb_state["state"]

            module = importlib.import_module(module_path)
            callback_cls = getattr(module, class_name)
            callbacks.append(callback_cls.load_state_dict(state_dict, **kwargs))

        # load the models.
        model_states = state["model_state_dict"]
        model_module = importlib.import_module(model_states["module"])
        model_cls = getattr(model_module, model_states["class_name"])
        model = model_cls.load(model_states["state"])

        # load the trainer
        trainer = cls(model, config, callbacks=callbacks, device=device)
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        trainer.global_step_ = state.get("global_step_", 0)

        return trainer