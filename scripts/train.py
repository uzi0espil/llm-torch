from dataclasses import asdict
from typing import Optional
import argparse
import tiktoken

from llm_torch.configs import LLMConfig
from llm_torch.data import create_dataloader
from llm_torch.engine import Trainer, Predictor
from llm_torch.architectures import Transformer
from llm_torch.utils import plot_losses

from scripts.configs import get as get_config


def train(data_path, model_config: LLMConfig, tokenizer,
          data_split=0.8, to_save: Optional[str] = None,
          device: str = "cuda"):

    with open(data_path) as f:
        text_data = f.read()

    train_ratio = data_split
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    dataset_config = model_config.dataset_config

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

    model = Transformer(model_config.model_config, model_config.vocab_size, model_config.context_length)

    callbacks = [callback.instantiate() for callback in model_config.callback_configs]
    trainer = Trainer(model, model_config.train_config, callbacks=callbacks, device=device)

    history = trainer.fit(train_dataloader, val_dataloader)

    plot_losses(history["loss"], history["val_loss"])

    if to_save:
        trainer.save(to_save)

    return Predictor(trainer.model, tokenizer, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--llm", type=str, required=True, help="Name of the language model to train.")
    parser.add_argument("--size", type=str, required=False, help="Size of the language model.",
                        default="124M")
    parser.add_argument("--data-split", type=float, default=0.8, help="Ratio of data to use for training.")
    parser.add_argument("--to-save", type=str, default=None, help="Path to save the trained model.")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device to train on.")

    parser.add_argument("--max-new-words", type=int, default=256,
                        help="Max number of new tokens to generate per prompt.")
    parser.add_argument("--temperature", "-t", type=float, required=False, default=1.,
                        help="Set the temperature of the prediction.")
    parser.add_argument("--top-k", "-k", type=int, required=False, default=None,
                        help="Consider the top k tokens")
    parser.add_argument("--use-cache", type=bool, required=False, default=True,
                        help="Speedup prediction using cache.")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")  # todo: temp. need to replace it with other tokenizer classes.
    config = get_config(args.llm, args.size)

    predictor = train(
        data_path=args.data_path,
        model_config=config,
        tokenizer=tokenizer,
        data_split=args.data_split,
        to_save=args.to_save,
        device=args.device,
    )

    while True:
        try:
            text = input("You: ")
            prediction = predictor.predict(text, args.max_new_words, temperature=args.temperature, top_k=args.top_k)
            print(f"{args.llm}: {prediction[0]}")
        except (KeyboardInterrupt, EOFError):
            print("Bye")
            break
