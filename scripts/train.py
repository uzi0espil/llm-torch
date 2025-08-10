from dataclasses import asdict
from typing import Optional
import argparse
import tiktoken

from llm_torch.configs import LLMConfig, get as get_config
from llm_torch.data import create_dataloader
from llm_torch.engine import Trainer, Predictor
from llm_torch.architectures import get as get_llm
from llm_torch.utils import plot_losses


def train(data_path, llm: str, config: LLMConfig, tokenizer,
          data_split=0.8, to_save: Optional[str] = None,
          device: str = "cuda"):

    with open(data_path) as f:
        text_data = f.read()

    train_ratio = data_split
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    dataset_config = config.dataset_config

    train_dataloader = create_dataloader(
        tokenizer,
        train_data,
        drop_last=True,
        num_workers=0,
        **asdict(dataset_config),
    )

    val_dataloader = create_dataloader(
        tokenizer,
        val_data,
        drop_last=False,
        num_workers=0,
        **asdict(dataset_config),
    )

    Llm = get_llm(name=llm)
    model = Llm(config.model_config, config.vocab_size, config.context_length)

    callbacks = [callback.instantiate() for callback in config.callback_configs]
    trainer = Trainer(model, config.train_config, callbacks=callbacks, device=device)

    history = trainer.fit(train_dataloader, val_dataloader)

    plot_losses(history["loss"], history["val_loss"])

    if to_save:
        trainer.save(to_save)

    return Predictor(trainer.model, tokenizer, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--llm", type=str, required=True, help="Name of the language model to train.")
    parser.add_argument("--size", type=str, required=False, help="Size of the language model.",
                        default="124")
    parser.add_argument("--data_split", type=float, default=0.8, help="Ratio of data to use for training.")
    parser.add_argument("--to_save", type=str, default=None, help="Path to save the trained model.")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to train on.")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")  # todo: temp. need to replace it with other tokenizer classes.
    config = get_config(args.llm, args.size)

    predictor = train(
        data_path=args.data_path,
        llm=args.llm,
        config=config,
        tokenizer=tokenizer,
        data_split=args.data_split,
        to_save=args.to_save,
        device=args.device,
    )
