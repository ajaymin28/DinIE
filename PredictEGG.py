import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import os
import uuid
# Dataset class
class EEGDataset:
    # Constructor
    def __init__(self, eeg_signals_path, eeg_splits_path, subset='train',subject=1, time_low=20,time_high=480, model_type="cnn", imagesRoot="./data/images/imageNet_images", preprocessin_fn=None):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'

        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type
        self.imagesRoot = imagesRoot

        splits = torch.load(eeg_splits_path)
        subset_indexes = splits["splits"][0][f"{subset}"]

        loaded = torch.load(eeg_signals_path)

        self.subsetData = []
        self.labels = []
        self.images = []

        class_labels = loaded["labels"]
        image_names = loaded['images']


        for sub_idx in subset_indexes:
            if subject!=0:
                # print(loaded['dataset'][sub_idx]['subject'])
                if loaded['dataset'][sub_idx]['subject']==subject:
                    # print(loaded['dataset'][sub_idx]['subject'])
                    self.subsetData.append(loaded['dataset'][sub_idx])
                    # self.labels.append(class_labels[loaded["dataset"][sub_idx]['label']])
                    self.labels.append(loaded["dataset"][sub_idx]['label'])
                    self.images.append(image_names[loaded["dataset"][sub_idx]['image']])
            else:
                sub_idx = int(sub_idx)
                # print(sub_idx)
                self.subsetData.append(loaded['dataset'][sub_idx])
                # self.labels.append(class_labels[loaded["dataset"][sub_idx]['label']])
                self.labels.append(loaded["dataset"][sub_idx]['label'])
                self.images.append(image_names[loaded["dataset"][sub_idx]['image']])


        # Compute size
        self.size = len(self.subsetData)

        self.preprocessin_fn = None
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        self.trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)


    # Get size
    def __len__(self):
        return self.size

    # Get itemsubset
    def __getitem__(self, i):
        # Process EEG
        # eeg = self.data[i]["eeg"].float().t()
        eeg = self.subsetData[i]["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high,:]

        if self.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,self.time_high-self.time_low)
        # Get label
        # label = self.data[i]["label"]
        label = self.labels[i]

        folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{folder_name}/{self.images[i]}.JPEG"


        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            image = self.preprocessin_fn(image)

        # Return
        return eeg, label, image
    
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, channels=128, n_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size*channels)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        lstm_out, hidden_out = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])

        return out
    

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_size),
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out
    

def findTopKsimilarities(dataset,query_seq,index=None, topK=5):
    cosine_similarities = []
    cosine_similarities_labels = []
    query_seq = query_seq.view(-1, query_seq.size(1))

    cos_1 = torch.nn.CosineSimilarity(dim=-1)
    s_index = 0
    for data in dataset:

        eeg, label, image = data

        eeg_flattened = eeg.view(-1, eeg.size(1)*eeg.size(2)).to(device)

        cosine_sim = cos_1(query_seq,eeg_flattened) 

        cosine_similarities.append(abs(cosine_sim.cpu().numpy()))
        cosine_similarities_labels.append(label.cpu().numpy())

            # cosine_similarities_labels_idx.append(s_index)
        s_index +=1
            # break

    
    cosine_similarities = np.array(cosine_similarities).T
    cosine_similarities_labels = np.array(cosine_similarities_labels).T
    idx = np.argpartition(cosine_similarities[0], -topK)[-topK:]

    cosine_similarities = cosine_similarities[0][idx]
    cosine_similarities_labels = cosine_similarities_labels[0][idx]

    return cosine_similarities,cosine_similarities_labels


import threading
import time

cosines = []
consines_labels = []
query_vector_queue = []
MainThreadExit = False

Total = 0
Correct = 0

count = 0
lock = threading.Lock()

if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=1,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--subject',
                        type=int,
                        default=1,
                        help='Subject Data to train')
    parser.add_argument('--dataset',
                        type=str,
                        default="./data/eeg/eeg_signals_raw_with_mean_std.pth",
                        help='Dataset to train')
    parser.add_argument('--dataset_split',
                        type=str,
                        default="./data/eeg/block_splits_by_image_all.pth",
                        help='Dataset split')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--custom_model_weights',
                        type=str,
                        default="./models/raw/FC__subject1_epoch_200.pth",
                        help='custom model weights')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)


    weights = ResNet50_Weights.DEFAULT
    resnet50_model = resnet50(weights=weights)


    SUBJECT = FLAGS.subject
    BATCH_SIZE = int(FLAGS.batch_size)
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.dataset

    LSTM_INPUT_FEATURES = 2048 # should be image features output.
    LSTM_HIDDEN_SIZE = 460  # should be same as sequence length

    # Load dataset
    dataset = EEGDataset(subset="train",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())
    test_dataset = EEGDataset(subset="train",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )


    # Remove the final classification (softmax) layer
    model = torch.nn.Sequential(*(list(resnet50_model.children())[:-1])) 
    model.eval()
    model.to(device)

    CustModel = CustomModel(input_size=(LSTM_INPUT_FEATURES),output_size=(LSTM_HIDDEN_SIZE*128))

    if os.path.exists(FLAGS.custom_model_weights):
        CustModel = torch.load(FLAGS.custom_model_weights)
        print(f"loaded custom weights: {FLAGS.custom_model_weights}")

    CustModel.to(device)

    if FLAGS.mode=="train":

        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(LSTM.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(CustModel.parameters(), lr=learning_rate)

        for epoch in range(EPOCHS):
            Losses = []
            for data in dataloader:
                eeg, label, image = data
                # print(eeg.size())

                with torch.no_grad():
                    # Forward pass through the model to get features
                    features = model(image.to(device))
                    # features = features.view(-1, 1, features.size(1))
                    # print(features.size())

                
                features = features.view(-1, features.size(1))
                # features = features.view(-1, features.size(1)*features.size(2)*features.size(3))

                # print("Image features ",features.size())

                optimizer.zero_grad()
                

                # outputs = LSTM(features)
                outputs = CustModel(features)

                eeg_flattened = eeg.view(-1, eeg.size(1)*eeg.size(2)).to(device)
                # eeg_flattened = eeg.view(BATCH_SIZE, -1).to(device)
                # print(eeg_flattened.size())

                # print("criterion input ", outputs.size(), eeg_flattened.size())
                loss = criterion(outputs, eeg_flattened)
                
                Losses.append(loss.item())
                
                
                loss.backward()
                optimizer.step()

            Losses = np.array(Losses)
            print(f"Epoch: {epoch} Loss: {Losses.mean()} ")

            if epoch%SaveModelOnEveryEPOCH == 0:
                torch.save(CustModel, '%s__subject%d_epoch_%d.pth' % ("FC", SUBJECT ,epoch))

        
        torch.save(CustModel, '%s__subject%d_Final.pth' % ("FC", SUBJECT))

    else:


        CustModel.eval()

        Correct = 1
        Total = 1

        index = 0
        for data in test_dataloader:
            eeg, label, image = data

            with torch.no_grad():
                features = model(image.to(device))
                features = features.view(-1, features.size(1))
                # print(CustModel)
                # print(features.shape)
                outputs = CustModel(features)

                with lock:
                    query_vector_queue.append([outputs,label.cpu().numpy()])

                cosine_similarities,cosine_similarities_labels = findTopKsimilarities(dataloader,outputs,index,topK=5)
                print(cosine_similarities,cosine_similarities_labels,label.cpu().numpy(),f"[{Total}/{Correct}][{round((Correct*100)/Total, 2)}]")
                if label.cpu().numpy() in cosine_similarities_labels:
                    Correct +=1

            Total +=1
            index +=1


                
