from torch.utils.data import DataLoader

from llm_torch.data.datasets import LLMDataset


def create_dataloader(tokenizer, txt, batch_size=4, max_length=256, stride=128, shuffle=True, **kwargs):
    # Create dataset
    dataset = LLMDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return dataloader
