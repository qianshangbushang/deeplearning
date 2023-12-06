import torch
import torch.nn as nn

import torchvision.models as models



class EncoderCNN(nn.Module):
    def __init__(self, embed_size, tran_cnn=False) -> None:
        super().__init__()
        self.train_cnn = tran_cnn
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        return
    
    def forward(self, images):
        if self.training:
            features, _ = self.inception(images)
        else:
            features = self.inception(images)
        return self.dropout(self.relu(features))
    


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings) # seq_len, N, hidden_size
        outputs = self.linear(hiddens)  # seq_len, N, vocab_size
        return outputs
    


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocba_size, num_layers):
        super().__init__()
        self.encoderCNN = EncoderCNN(embed_size,)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocba_size, num_layers)
        return
    
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def caption_iamge(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            stats = None

            for _ in range(max_length):
                hiddens, stats = self.decoderRNN.lstm(x, stats)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
            print(result_caption)
            return [vocabulary.itos[idx] for idx in result_caption]