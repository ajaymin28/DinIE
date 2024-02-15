import torch
import torch.nn as nn
import torchvision.models as models
from utils.EEGDataset import EEGDataset
import argparse
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class ResnetFeatureRegressor(nn.Module):
    def __init__(self, num_features, output_size):
        super(ResnetFeatureRegressor, self).__init__()
        
        # Load a pre-trained CNN as the feature extractor
        weights = ResNet50_Weights.DEFAULT
        self.cnn = resnet50(weights=weights)
        
        # Remove the classification head of the CNN
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze the weights of the CNN
        for param in self.cnn.parameters():
            param.requires_grad = True
        
        # Define a regression head
        self.fc = nn.Linear(num_features, output_size)

        self.preprocessin_fn = weights.transforms()
    
    def forward(self, x):
        # Extract features using the CNN
        features = self.cnn(x)
        
        # Flatten the features
        features = torch.flatten(features, 1)
        
        # Regression
        output = self.fc(features)
        return output




if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
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
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    

    # Example usage
    num_features = 2048  # Number of features output by the CNN
    output_size = 460 * 128  # should be same as sequence length  # Size of the feature vector to regress

    # Create an instance of the model
    model = ResnetFeatureRegressor(num_features, output_size)
    model = model.to(device=device)


    # Load dataset
    dataset = EEGDataset(subset="train",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=model.preprocessin_fn)
    test_dataset = EEGDataset(subset="test",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=model.preprocessin_fn)
    val_dataset = EEGDataset(subset="val",eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=model.preprocessin_fn)


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

    
    import os
    os.makedirs(FLAGS.log_dir, exist_ok=True)


    if FLAGS.mode=="train":

        criterion = nn.SmoothL1Loss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)

        All_Losses = []
        Val_All_losses = []
        Test_All_losses = []
        for epoch in range(EPOCHS):

            Losses = []
            Val_losses = []
            model.train()

            for data in dataloader:
                eeg, label, image, i = data

                eeg_flattened = eeg.view(eeg.size(0), -1).to(device)
                image = image.to(device)

                optimizer.zero_grad()

                outputs = model(image)

                loss = criterion(outputs, eeg_flattened)

                Losses.append(loss.item())

                loss.backward()
                optimizer.step()
            
            Losses = np.array(Losses)
            All_Losses.append(Losses)


            # Validate the model
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_dataloader:
                    eeg, label, image, i = data
                    eeg_flattened = eeg.view(eeg.size(0), -1).to(device)
                    image = image.to(device)
                    outputs = model(image)
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
                    outputs = model(image)
                    test_loss += criterion(outputs, eeg_flattened).item()
            test_loss /= len(test_dataloader)
            Test_All_losses.append(test_loss)


            print(f"Epoch: {epoch} Loss: {Losses.mean()} Val Loss: {val_loss:.2f} Test loss: {test_loss:.2f}")

            if epoch%SaveModelOnEveryEPOCH == 0:
                torch.save(model, '%s/%s__subject%d_epoch_%d.pth' % (FLAGS.log_dir,"Resnet_finetuned_eeg", SUBJECT ,epoch))
        

        torch.save(model, '%s/%s__subject%d_Final.pth' % (FLAGS.log_dir,"Resnet_finetuned_eeg", SUBJECT))
        torch.save(All_Losses, '%s/%s__subject%d_Final_All_losses.pth' % (FLAGS.log_dir,"Resnet_finetuned_eeg", SUBJECT))
        torch.save(Val_All_losses, '%s/%s__subject%d_Final_Val_All_losses.pth' % (FLAGS.log_dir,"Resnet_finetuned_eeg", SUBJECT))
        torch.save(Test_All_losses, '%s/%s__subject%d_Final_Test_All_losses.pth' % (FLAGS.log_dir,"Resnet_finetuned_eeg", SUBJECT))
    # # Forward pass
    # output_feature_vector = model(input_image)
    # print(output_feature_vector.size())  # Should print torch.Size([1, 10])
