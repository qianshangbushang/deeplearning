
import torch

model_name = "video_caption.pth.tar"

def save_checkpoint(ckpt: dict):
    torch.save(ckpt, model_name)
    return


def load_checkpoint():
    model = torch.load(model_name, )
    return model