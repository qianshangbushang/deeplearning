import torch
import torch.nn as nn
from torchvision import transforms

import torch.optim as optim
from model import CNNtoRNN
from data import get_loader
from util import save_checkpoint, load_checkpoint
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def run(root_folder, caption_file, load_pretrained=False, save_model=False, emb_size=256,  hidden_size=128, num_layers=1, lr=3e-4, max_epoch=100, device='cpu'):
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

    writer = SummaryWriter("runs/flicker")
    vocab_size = data_loader.dataset.vocab_size()
    model = CNNtoRNN(emb_size, hidden_size, vocab_size, num_layers)
    if load_pretrained:
        model.load_state_dict(load_checkpoint())


    model = model.to(device)
    optimmizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=data_loader.dataset.vocabulary.stoi["<PAD>"])

    for name, param in model.encoderCNN.inception.named_parameters():
        if 'fc.weight' in name or 'fc.bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train()

    for epoch in range(max_epoch):
        if save_model:
            save_checkpoint(model.state_dict())

        for batch_idx, (imgs, captions) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimmizer.zero_grad()
            loss.backward(loss)
            optimmizer.step()

    save_checkpoint(model.state_dict())
    print("finished")
    return


if __name__ == "__main__":
    root_folder = "./flicker8k/images"
    caption_file = "./flicker8k/captions.txt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run(root_folder, caption_file, device=device)
