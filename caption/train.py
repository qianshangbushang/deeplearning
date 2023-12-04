import torch
import torch.nn as nn
from torchvision import transforms

import torch.optim as optim
from model import CNNtoRNN
from data import get_loader
from util import save_checkpoint, load_checkpoint


def run(root_folder, caption_file, load_pretrained=False, save_model=False, emb_size=256,  hidden_size=128, num_layers=3, lr=3e-4, max_epoch=100, device='cpu'):
    transfomr = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_loader = get_loader(
        root_folder=root_folder,
        annotation_file=caption_file,
        transform=transfomr,
        num_workers=2,
    )

    vocab_size = data_loader.dataset.vocab_size()
    model = CNNtoRNN(emb_size, hidden_size, vocab_size, num_layers)
    if load_pretrained:
        model.load_state_dict(load_checkpoint())

    optimmizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=data_loader.dataset.vocabulary.stoi["<PAD>"])

    for epoch in range(max_epoch):
        if save_model:
            check_point = model.state_dict()

        for batch_idx, (imgs, captions) in enumerate(data_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimmizer.zero_grad()
            loss.backward()
            optimmizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch}, batch: {batch_idx} loss: {loss.item()}")

    if save_model:
        save_checkpoint(model.state_dict())
    print("finished")
    return


if __name__ == "__main__":
    root_folder = "./flicker8k/images"
    caption_file = "./flicker8k/captions.txt"
    run(root_folder, caption_file)
