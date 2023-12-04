import os

import torch
import torch.nn as nn

from typing import Any
import pandas as pd
import spacy

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from collections import defaultdict
from PIL import Image

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_token = "<SOS>"
end_token = "<EOS>"
pad_token = "<PAD>"
unknown_token = "<UNK>"

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold) -> None:
        self.freq_threshold = freq_threshold
        self.stoi = {pad_token:0, start_token:1, end_token:2, unknown_token:3}
        self.itos = {v: k for k, v in self.stoi.items()}
        return
    
    def __len__(self):
        return len(self.stoi)

    @staticmethod   
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    
    def build_vocabulary(self, sentence_list):
        frequence_dict = defaultdict(int)
        word_index = max(self.stoi.values()) + 1
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequence_dict[word] += 1
                if frequence_dict[word] == self.freq_threshold:
                    self.stoi[word] = word_index
                    self.itos[word_index] = word
                    word_index += 1
        return 
    
    def numericalize(self, sentence):
        return [
            self.stoi[token] if token in self.stoi else self.stoi[unknown_token]
            for token in self.tokenizer_eng(sentence)
        ]

class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(captions_file)

        self.images = self.df["image"]
        self.captions = self.df["caption"]

        self.vocabulary = Vocabulary(freq_threshold)
        self.vocabulary.build_vocabulary(self.captions.tolist())
        return
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, self.images[index])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        numericalize_caption = [self.vocabulary.stoi[start_token]]
        numericalize_caption += self.vocabulary.numericalize(self.captions[index])
        numericalize_caption += [self.vocabulary.stoi[end_token]]
        return img, torch.tensor(numericalize_caption)
    
    def vocab_size(self):
        return len(self.vocabulary)



class MyCollate:
    def __init__(self, pad_idx) -> None:
        self.pad_idx = pad_idx
        return
    
    def __call__(self, batch) -> Any:
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        tgts = [item[1] for item in batch]
        tgts = pad_sequence(tgts, batch_first=False, padding_value=self.pad_idx)
        return imgs, tgts


def get_loader(
        root_folder,
        annotation_file,
        transform,
        freq_threshold=5,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = FlickerDataset(root_folder, annotation_file, transform, freq_threshold)
    pad_idx = dataset.vocabulary.stoi[pad_token]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataloader = get_loader("flicker8k/images/", annotation_file="flicker8k/captions.txt", transform=transform)
    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)
