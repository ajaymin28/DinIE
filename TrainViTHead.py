import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as transforms 

from utils.CustomModel import CustomModel
from utils.EEGDataset import EEGDataset
import argparse
import os
import numpy as np

# Define the dense layer for regression
class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionHead, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.dense(x)


if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=8,
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
                        default="",
                        help='custom model weights')
    parser.add_argument('--query_dataset',
                    type=str,
                    default="EEG",
                    help='EEG,caltech101')
    parser.add_argument('--search_gallary',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--query_gallary',
                        type=str,
                        default="domainnet",
                        help='dataset in which images will be searched')
    parser.add_argument('--domainnet_subtype',
                        type=str,
                        default="clipart",
                        help='Sub type of Domainnet dataset')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    
    parser.add_argument('--class_to_search',
                        type=str,
                        default="",
                        help='dataset class to search ')
    
    parser.add_argument('--imagenet_label_name',
                        type=str,
                        default="",
                        help='imagenet label class name')


    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    
    SUBJECT = FLAGS.subject
    BATCH_SIZE = int(FLAGS.batch_size)
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 10
    EEG_DATASET_PATH = FLAGS.dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    


    # Define the ViT feature extractor and model
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # # Create an instance of the model
    # model = ResnetFeatureRegressor(num_features, output_size)
    vit_model = vit_model.to(device=device)

    transform = transforms.Compose([ 
            # transforms.PILToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 


    # Load dataset
    dataset = EEGDataset(subset="train",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=transform)
    test_dataset = EEGDataset(subset="test",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=transform)
    val_dataset = EEGDataset(subset="val",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=transform)


    # # Example input image
    # input_image = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image of size 224x224

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

    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )



    # Example usage
    num_input_features = 2048  # Number of features output by the CNN
    output_size = 460 * 128  # should be same as sequence length  # Size of the feature vector to regress

    for data in dataloader:
        eeg, label, image, i = data
        image = image.to(device)
        with torch.no_grad():            
            features = vit_model(image).last_hidden_state[:, 0]  # Extract features from the last hidden state
            # features_flat = features.flatten(start_dim=1)
            print("features size::", features.size(), "features_flat::", features.size())
            num_input_features  = features.size(1)
        break


    CustModel = CustomModel(input_size=num_input_features,output_size=output_size)
    if os.path.exists(FLAGS.custom_model_weights):
        CustModel = torch.load(FLAGS.custom_model_weights)
        print(f"loaded custom weights: {FLAGS.custom_model_weights}")

    CustModel.to(device)


    OutputDir = f"{FLAGS.log_dir}/{SUBJECT}"
    os.makedirs(OutputDir, exist_ok=True)


    if FLAGS.mode=="train":

        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(LSTM.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(CustModel.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)


        All_Losses = []
        Val_All_losses = []
        Test_All_losses = []
        for epoch in range(EPOCHS):

            CustModel.train()

            Losses = []
            for data in dataloader:
                eeg, label, image, i = data

                # eeg_flattened = eeg.view(eeg.size(0), -1).to(device)
                image = image.to(device)

                with torch.no_grad():
                    features = vit_model(image).last_hidden_state[:,0]  # Extract features from the last hidden state
                    # features_flat = features.flatten(start_dim=1)

                optimizer.zero_grad()

                outputs = CustModel(features)
                eeg_flattened = eeg.view(eeg.size(0), -1).to(device)

                loss = criterion(outputs, eeg_flattened)
                
                Losses.append(loss.item())
                
                loss.backward()
                optimizer.step()

            
            Losses = np.array(Losses)
            # print(f"Epoch: {epoch} Loss: {Losses.mean()} ")

            # Validate the model
            CustModel.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_dataloader:
                    eeg, label, image, i = data
                    eeg_flattened = eeg.view(eeg.size(0), -1).to(device)
                    image = image.to(device)
                    features = vit_model(image).last_hidden_state[:,0]  # Extract features from the last hidden state
                    outputs = CustModel(features)
                    val_loss += criterion(outputs, eeg_flattened).item()

            val_loss /= len(val_dataloader)
            Val_All_losses.append(val_loss)
            # Update learning rate
            scheduler.step(val_loss)

            
            test_loss = 0.0
            with torch.no_grad():
                for data in test_dataloader:
                    eeg, label, image, i = data
                    eeg_flattened = eeg.view(eeg.size(0), -1).to(device)
                    image = image.to(device)
                    features = vit_model(image).last_hidden_state[:,0]  # Extract features from the last hidden state
                    outputs = CustModel(features)
                    test_loss += criterion(outputs, eeg_flattened).item()
            test_loss /= len(test_dataloader)
            Test_All_losses.append(test_loss)


            print(f"Epoch: {epoch} Loss: {Losses.mean():.2f} Val Loss: {val_loss:.2f} Test loss: {test_loss:.2f}")

            if epoch%SaveModelOnEveryEPOCH == 0:
                torch.save(CustModel, f"{OutputDir}/VIT_Head_finetuned_eeg_subject_{SUBJECT}_epoch{epoch}.pth")


            # optimizer.zero_grad()

            # outputs = model(image)

            # loss = criterion(outputs, eeg_flattened)

            # Losses.append(loss.item())

            # loss.backward()
            # optimizer.step()




        # # Example input image (224x224)
        # input_image = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels (RGB), 224x224 resolution

        # # Extract features using ViT model
        # features = model(input_image).last_hidden_state  # Extract features from the last hidden state

        # # Flatten the features
        # features_flat = features.flatten(start_dim=1)

        # # Define the number of output nodes for regression
        # output_dim = 1  # Example: Regression output dimension

        # # Initialize regression head
        # regression_head = RegressionHead(features_flat.size(1), output_dim)

        # # Perform regression
        # output = regression_head(features_flat)

        # print("Output shape:", output.shape)  # Example: Output shape: torch.Size([1, 1])
