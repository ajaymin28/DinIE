from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from imutils import paths
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import time
from tqdm import tqdm
# custom dataset class
class ImageNetDatasetImgRet(Dataset):
    def __init__(self, images_path, preprocessin_fn = None, filter_label=None, test_split=0.2, random_seed=43, subset="train", n_classes=1000):

        
        imagenetClaases_to_exclude = []
        # data\images\imageNet_images\40classLabels.txt
        with open(f"/lustre/fs1/home/jbhol/EEG/mytraining/data/images/imageNet_images/40classLabels.txt") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.strip()
                imagenetClaases_to_exclude.append(line)

        print(f"Classes {imagenetClaases_to_exclude} will be exluded")
        
        
        imagenetClaases_to_include = []
        # data\images\imageNet_images\40classLabels.txt
        with open(f"/lustre/fs1/home/jbhol/EEG/mytraining/data/images/imageNet_images/ImagenetClassLabelsToTest.txt") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.strip()
                imagenetClaases_to_include.append(line)

        print(f"Classes {imagenetClaases_to_include} will be included")
        
        
        image_paths = list(paths.list_images(images_path))

        self.str_labels = []
        self.images = []
        self.EEGs = []
        self.image_features = []

        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label in imagenetClaases_to_exclude:
                continue
            if label in imagenetClaases_to_include:
                self.images.append(img_path.replace("\\", "/"))
                self.str_labels.append(label)

        lb = LabelEncoder()
        self.labels = lb.fit_transform(self.str_labels)
        print(self.labels)

        self.class_str_to_id = {}
        for intlab, strlab in zip(self.labels,self.str_labels):
            if strlab not in self.class_str_to_id:
                self.class_str_to_id[strlab] = intlab

        self.int_to_str_labels = {y: x for x, y in self.class_str_to_id.items()}

        # Create a StratifiedShuffleSplit instance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_seed)

        # Get the indices for the splits
        for train_index, test_index in sss.split(self.images, self.labels):
            self.train_images = [self.images[i] for i in train_index]
            self.train_labels = [self.labels[i] for i in train_index]
            self.test_images = [self.images[i] for i in test_index]
            self.test_labels = [self.labels[i] for i in test_index]

        
        print(len(set(self.test_labels)), len(set(self.train_labels)))
        assert len(set(self.test_labels))==len(set(self.train_labels))

        if subset == "train":
            self.images = self.train_images
            self.labels = self.train_labels
        elif subset == "test":
            self.images = self.test_images
            self.labels = self.test_labels

        self.preprocessin_fn = preprocessin_fn
        self.isDataTransformed = False
        self.isImageFeaturesExtracted = False

    def getOriginalImage(self, idx):
        ImagePath = self.images[idx][:]
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        ImagePath = self.images[idx][:]
        return ImagePath
    

    def resizeEEGToImageSize(self,input_data=None,imageShape=(224,224),index=None):

        if input_data is None:
            if index is None:
                raise Exception("Need either input data or give dataset index")
            else:
                input_data = self.EEGs[index].float()
                input_data = input_data.cpu().numpy()
        # print("EE sample size",self.subsetData[i]["eeg"].size())
        # eeg = self.subsetData[i]["eeg"].T
        if not self.isDataTransformed:
            input_data = input_data[self.time_low:self.time_high,:]


        # print(input_data.shape)
        IMG_H, IMG_W  = imageShape[0], imageShape[1]
        # EEG input_data is assumed to be a numpy array of shape (128, 460)

        # Repeat each channel until we reach IMG_H channels
        repeated_data = np.repeat(input_data, (IMG_H // input_data.shape[0])+1, axis=0)
        repeated_data = np.repeat(repeated_data, (IMG_W // repeated_data.shape[1])+1, axis=1)

        # print("repeated_data: ",repeated_data.shape)

        # If we have more than IMG_H channels, slice the array down to IMG_H
        if repeated_data.shape[0] > IMG_H:
            repeated_data = repeated_data[:IMG_H, :]

        # print("reduced height repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, 460). We need to slice the time series data to get IMG_H x IMG_W.

        # If we have more than IMG_W time series data points, slice the array down to IMG_W
        if repeated_data.shape[1] > IMG_W:
            start_index = np.random.randint(0, repeated_data.shape[1]-IMG_W)
            repeated_data = repeated_data[:, start_index:start_index+IMG_W]
            # print(f"start:{start_index} end: {start_index+224}")
            # repeated_data = repeated_data[:, :IMG_W]d

        # print("reduced width repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, IMG_W). We need to repeat this for 3 color channels to get IMG_HxIMG_Wx3.

        # Repeat the 2D array along a new third dimension
        output_data = np.repeat(repeated_data[np.newaxis, :, :], 3, axis=0)

        """
        Z2-score Normlization  https://arxiv.org/pdf/2210.01081.pdf
        """
        # fmean = np.mean(output_data)
        # fstd = np.std(output_data)
        # output_data = (output_data - fmean)/fstd

        # print("increased c: ",output_data.shape)

        return output_data
    
    def ExtractImageFeatures(self, preprocsessor, model, device):
        print("Extracting Image features")
        for img_idx in range(len(self.images)):
            image =  self.getOriginalImage(img_idx)
            with torch.no_grad():
                inputs = preprocsessor(images=image, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                features = last_hidden_states[:,0,:] # batch size, 257=CLS_Token+256,features_length
                features = features.reshape(features.size(0), -1)
                self.image_features.append(features)
        print("Extracting Image features done")

        if len(self.image_features)==len(self.images):
            self.isImageFeaturesExtracted = True
        else:
            print("Image features are extracted but their lenght doesnt match with total images.")
        
    
    def transformEEGDataDino(self, model, device, pass_eeg=False,preprocessor=None, min_time=0,max_time=460, do_z2_score_norm=False, keep_features_flat=False):
        print(f"Transforming Image data to dino features EEG, pass_eeg is {pass_eeg}")
        # assert pass_eeg==False
        model = model.to(device)

        for i, image_path in tqdm(enumerate(self.images), total=len(self.images)):
            t0 = time.perf_counter()
            model_inputs = None
            if pass_eeg:
                eeg = self.EEGs[i].float()
                eeg = eeg.cpu().numpy()
                eeg = self.resizeEEGToImageSize(eeg)
                if do_z2_score_norm:
                    fmean = np.mean(eeg)
                    fstd = np.std(eeg)
                    eeg = (eeg - fmean)/fstd
                model_inputs = torch.from_numpy(eeg)
            else:

                model_inputs = self.getOriginalImage(i)

                if self.preprocessin_fn is not None:
                    model_inputs = self.preprocessin_fn(model_inputs)
                else:
                    if preprocessor is not None:
                        model_inputs = preprocessor(model_inputs)

            with torch.no_grad():
                feats = model(model_inputs.unsqueeze(0).to(device))
                if not keep_features_flat:
                    dino_f = feats.cpu().numpy()
                    dino_f = dino_f.reshape(128, -1)
                    dino_f = dino_f[:,min_time:max_time]
                    self.EEGs.append(torch.from_numpy(dino_f).float())
                else:
                    self.EEGs.append(feats.float())

        self.isDataTransformed = True
        print("Transforming Image data to dino EEG features (done)")

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        returnData =  self.getOriginalImage(idx)
        if self.preprocessin_fn is not None:
            returnData = self.preprocessin_fn(returnData)

        if self.isDataTransformed:
            returnData = self.EEGs[idx]
        return returnData, idx