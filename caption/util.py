
import torch

model_name = "../checkpoint/video_caption.pth.tar"

def save_checkpoint(ckpt: dict):
    torch.save(ckpt, model_name)
    return


def load_checkpoint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_name, map_location=device)
    return model