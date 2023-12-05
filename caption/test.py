import torch

import torch.nn as nn
from model import CNNtoRNN
from data import get_loader
from torchvision import transforms
from util import load_checkpoint


def test_image():
    root_folder = "./flicker8k/images"
    caption_file = "./flicker8k/captions.txt"
    transfomr = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loader = get_loader(
        root_folder=root_folder,
        annotation_file=caption_file,
        transform=transfomr,
        num_workers=2,
        batch_size=1
    )
    vocab_size = data_loader.dataset.vocab_size()
    model = CNNtoRNN(embed_size=256,  hidden_size=128,
                     num_layers=3, vocba_size=vocab_size)

    model.load_state_dict(load_checkpoint())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    vocabulary = data_loader.dataset.vocabulary
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        pred = " ".join(model.caption_iamge(imgs, vocabulary, 50))
        true = " ".join([vocabulary.itos[idx.item()] for idx in captions])
        print("true: ", true)
        print("pred: ", pred)

        if batch_idx == 10:
            break


if __name__ == "__main__":
    test_image()
