# imports
import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from vocabulary import Vocabulary, spacy_eng

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pickle

vocab=pickle.load(open('model/vocabulary.pkl','rb'))
print(len(vocab.itos))
#custom imports
from custom_utils import load_checkpoint,save_checkpoint 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        #self.inception.eval()
        #print(self.inception)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
       # print("images.shape",images.shape)
        features = self.inception(images)
        #p#rint("features.shape",features.shape)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        #print("captions",captions)
        embeddings = self.dropout(self.embed(captions))
        #print("caption_embeddings.shape",embeddings.shape)
        #print("features.unsqueeze(0).shape",features.unsqueeze(0).shape)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]



def print_examples(model, device, vocab,image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    img1=transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    return " ".join(model.caption_image(img1.to(device), vocab))



embed_size = 256
hidden_size = 256
vocab_size = len(vocab)
num_layers = 1
learning_rate = 3e-4
#num_epochs = 50

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
step = load_checkpoint(torch.load("model/my_checkpoint.pth.tar",map_location=torch.device("cpu")), model, optimizer)

import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


#gradio
import gradio as gr

def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # remove the file
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

def greet(video_file):
    basename='testvideo_img'
    dir_path='test'
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    cv2.destroyAllWindows()

    save_all_frames(video_file,dir_path,basename)
    list_img=os.listdir(dir_path)
    print(list_img[0:10],len(list_img))
    step=int(len(list_img )//10)+1
    caption_list=[]
    frame_time=[]
    for i in range(0,len(list_img),step):
        caption_list.append(print_examples(model,device,vocab,os.path.join(dir_path,list_img[i])))
        frame_time.append(int(i/fps))

    caption_list1=[i.replace('<EOS>',' ') for i in caption_list]
    caption_list=[i.replace('<SOS>',' ') for i in caption_list1]
    print('caption list:',caption_list)
    caption_list1=[]
    for i in range(0,len(caption_list)):
        caption_list1.append(str(frame_time[i])+"s:"+ caption_list[i]+"\n")
    delete_files(dir_path)
    
    return ' '.join(caption_list1)


if __name__=="__main__":
    demo=gr.Interface(fn=greet,inputs='video',outputs='text')

    demo.launch()