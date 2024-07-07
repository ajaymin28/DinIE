import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import random
class EEGDataset(Dataset):
    # Constructor
    def __init__(self, eeg_signals_path, 
                 eeg_splits_path, 
                 subset='train',
                 subject=1, 
                 exclude_subjects=[],
                 filter_channels=[],
                 time_low=20,time_high=480, 
                 model_type="cnn", 
                 imagesRoot="./data/images/imageNet_images",
                 apply_norm_with_stds_and_means=False,
                 apply_channel_wise_norm=False,
                 preprocessin_fn=None,
                 Transform_EEG2Image_Shape=False,
                 convert_image_to_tensor=False,
                 perform_dinov2_global_Transform=False,
                 dinov2Config=None,
                 inference_mode=True,
                 onehotencode_label=False,
                 data_augment_eeg=False,
                 add_channel_dim_to_eeg=False):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'


        self.Transform_EEG2Image_Shape = Transform_EEG2Image_Shape
        self.convert_image_to_tensor = convert_image_to_tensor
        self.perform_dinov2_global_Transform = perform_dinov2_global_Transform
        self.dinov2Config = dinov2Config
        self.apply_norm_with_stds_and_means = apply_norm_with_stds_and_means
        self.apply_channel_wise_norm = apply_channel_wise_norm
        self.filter_channels = filter_channels
        self.inference_mode = inference_mode
        self.onehotencode_label = onehotencode_label
        self.data_augment_eeg = data_augment_eeg
        self.add_channel_dim_to_eeg = add_channel_dim_to_eeg

        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type
        self.imagesRoot = imagesRoot

        # splits = torch.load(eeg_splits_path)
        # subset_indexes = splits["splits"][0][f"{subset}"]

        loaded = torch.load(eeg_signals_path)

        self.subsetData = []
        self.labels = []
        self.images = []
        self.image_features = []

        self.class_labels = loaded["labels"]
        image_names = loaded['images']

        std_channel_wise = loaded["stddevs"][0]
        mean_channel_wise = loaded["means"][0]

        EEGSelectedImageNetClasses = []
        for imageP in image_names:
            class_folder_name = imageP.split("_")[0]
            EEGSelectedImageNetClasses.append(class_folder_name)

        self.class_labels_names = {}
        self.class_id_to_str = {}
        self.class_str_to_id = {}

        lines = []
        with open(f"{imagesRoot}/labels.txt") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.strip()
                line = line.split(" ")
                imagenetDirName = line[0]
                imagenetClassId = line[1]
                imagenetClassName = line[-1]
                if imagenetDirName in EEGSelectedImageNetClasses:
                    indexOfClass = self.class_labels.index(imagenetDirName)
                    self.class_labels_names[imagenetDirName] = {"ClassId": int(indexOfClass), "ClassName": imagenetClassName, "imagenetClassId": imagenetClassId}
                    self.class_id_to_str[int(indexOfClass)]= imagenetClassName
                    self.class_str_to_id[imagenetClassName]= int(indexOfClass)
        
        self.std = 0
        self.mean = 0
        cnt = 0
        
        for i in range(len(loaded["dataset"])):
            eeg = loaded['dataset'][i]["eeg"].t()
            self.mean += eeg.mean()
            self.std  += eeg.std()
            cnt +=1
            if apply_norm_with_stds_and_means:
                eeg = (eeg-mean_channel_wise)/std_channel_wise
                loaded['dataset'][i]["eeg"] = eeg.t()
            self.subsetData.append(loaded['dataset'][i])
            self.labels.append(loaded["dataset"][i]['label'])
            self.images.append(image_names[loaded["dataset"][i]['image']])

        self.mean = self.mean/cnt
        self.std = self.std/cnt

        # Compute size
        self.size = len(self.subsetData)

        self.preprocessin_fn = None
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        self.trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
        
        
        if self.perform_dinov2_global_Transform:
            self.geometric_augmentation_global = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.dinov2Config.crops.global_crops_size, scale=self.dinov2Config.crops.global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

        self.isDataTransformed = False
        self.image_features_extracted = False

        if self.apply_channel_wise_norm:
            self.transformEEGDataToChannelWiseNorm()

        self.label_wise_data = {}
        #print("SELF LEN ",len(self))
        for idx in range(len(self)):
            class_folder_name = self.images[idx].split("_")[0]
            label = self.class_labels_names[class_folder_name]
            if not label["ClassId"] in self.label_wise_data:
                self.label_wise_data[label["ClassId"]] = {"indexes":[]}
            self.label_wise_data[label["ClassId"]]["indexes"].append(idx)
            #print(self.label_wise_data)

    def __len__(self):
        return self.size

    def generate_eeg_noise_data(self, num_channels, num_samples, sampling_rate=1000, frequency=40, amplitude = 0.5):
        # Step 1: Generate Gaussian noise
        gaussian_noise = np.random.normal(0, 1, size=(num_channels, num_samples))
        # Example: Add a sinusoidal oscillation to each channel
        time = np.arange(num_samples) / sampling_rate
        eeg_data = gaussian_noise + amplitude * np.sin(2 * np.pi * frequency * time)

        return eeg_data
    
    def transformToEEGNoisyData(self):
        print("Transforming EEG data to noisy data")
        for i, image_path in enumerate(self.images):
            eeg_noisy_data = self.generate_eeg_noise_data(num_channels=128,num_samples=500,sampling_rate=1000)
            self.subsetData[i]["eeg"] = torch.from_numpy(eeg_noisy_data).float()
        self.isDataTransformed = True
        print("Transforming EEG data (done)")
    
    def getOriginalImage(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        return ImagePath
    

    def resizeEEGToImageSize(self,input_data=None,imageShape=(224,224,3),min_channels=224,index=None):

        if input_data is None:
            if index is None:
                raise Exception("Need either input data or give dataset index")
            else:
                input_data = self.subsetData[index]["eeg"].float().t()
                input_data = input_data.cpu().numpy().T
        # print("EE sample size",self.subsetData[i]["eeg"].size())
        # eeg = self.subsetData[i]["eeg"].T
        if not self.isDataTransformed:
            input_data = input_data[self.time_low:self.time_high,:]


        # print(input_data.shape)
        IMG_H, IMG_W, IMG_C  = imageShape[0], imageShape[1], imageShape[2]
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
            
        
        #np.random.shuffle(repeated_data)  # randomly shuffle the channels, added on 18th April 2024, shuffling is done on first axis, so make sure first dim is channel
        
        #Randonmly select 32 channels
        #NumOfChannelsToKeep = np.random.randint(min_channels, 128)
        #random_indices = np.random.choice(128, size=min_channels, replace=False) # replace false means, once channel is picked up wont be considered for selection again. 
        #repeated_data = repeated_data[random_indices]

        # print("reduced width repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, IMG_W). We need to repeat this for 3 color channels to get IMG_HxIMG_Wx3.

        # Repeat the 2D array along a new third dimension
        output_data = np.repeat(repeated_data[np.newaxis, :, :], IMG_C, axis=0)
        
        """
        Z2-score Normlization  https://arxiv.org/pdf/2210.01081.pdf
        """
        #fmean = np.mean(output_data)
        #fstd = np.std(output_data)
        #output_data = (output_data - fmean)/fstd

        # print("increased c: ",output_data.shape)

        return output_data
    
    def ExtractImageFeatures(self, preprocsessor, model, device):
        print("Extracting Image features")
        for img_idx in tqdm(range(len(self.images)), total=len(len(self.images))):
            class_folder_name = self.images[img_idx].split("_")[0]
            ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[img_idx]}.JPEG"
            image = Image.open(ImagePath).convert('RGB')
            with torch.no_grad():
                inputs = preprocsessor(images=image, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                features = last_hidden_states[:,0,:] # batch size, 257=CLS_Token+256,features_length
                features = features.reshape(features.size(0), -1)
                self.image_features.append(features)
        print("Extracting Image features done")
        self.image_features_extracted = True
    

    
    def transformEEGData(self, resnet_model, resnet_to_eeg_model, device, isVIT=False):
        print("Transforming EEG data")
        for i, image_path in enumerate(self.images):

            class_folder_name = image_path.split("_")[0]
            ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

            image = Image.open(ImagePath).convert('RGB')
            if self.preprocessin_fn is not None:
                image = self.preprocessin_fn(image)

            # eeg, label, image, idxs = data
            with torch.no_grad():
                if isVIT:
                    features = resnet_model(image.unsqueeze(0).to(device)).last_hidden_state[:, 0]
                else:
                    features = resnet_model(image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                outputs = resnet_to_eeg_model(features)
                self.subsetData[i]["eeg"] = outputs
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        print("Transforming EEG data (done)")

    def getLabel(self, i):
        class_folder_name = self.images[i].split("_")[0]
        #ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"
        label = self.class_labels_names[class_folder_name]
        return label
    
    def getNegativeSampleIndex(self, currentSampleIndex):
        current_label = self.getLabel(i=currentSampleIndex)
        classes_index = list(self.label_wise_data.keys())
        filtered_classes_list = [cls for cls in classes_index if cls != current_label["ClassId"]]
        random_class = random.choice(filtered_classes_list)
        random_class_sample_idx = random.choice(self.label_wise_data[random_class]["indexes"])
        i  = random_class_sample_idx
        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"
        label = self.class_labels_names[class_folder_name]
        eeg, label,image,i, image_features = None,None,None,i,None
        image = Image.open(ImagePath).convert('RGB')
        return eeg, label,image,i, image_features
    
    def __getitem__(self, i):
        eeg = self.subsetData[i]["eeg"].float()

        if not self.isDataTransformed:
            if len(self.filter_channels)>0:
                eeg = eeg.cpu().numpy()
                eeg_zeros = np.zeros((self.time_high-self.time_low, len(self.filter_channels)), dtype=eeg.dtype)
                # print(f"eeg shape: {eeg.shape} eeg zeros shape: {eeg_zeros.shape}")
                for ch_idx, ch in enumerate(self.filter_channels):
                    # print(f" sliced: {eeg[self.time_low:self.time_high,ch].shape}")
                    eeg_zeros[:,ch_idx] = eeg[self.time_low:self.time_high,ch]
                    if self.apply_channel_wise_norm:
                        eeg_zeros = self.normlizeEEG(EEG=eeg_zeros,ch_index=ch_idx,class_index=None)
                eeg = torch.from_numpy(eeg_zeros).t()
                # print(f"final EEG : {eeg.size()}")
            else:
                # if self.apply_channel_wise_norm:
                #     for ch_idx in range(eeg.shape[-1]):
                #         eeg = self.normlizeEEG(EEG=eeg,ch_index=ch_idx,class_index=None)
                eeg = eeg[self.time_low:self.time_high,:]


        if self.Transform_EEG2Image_Shape:
            eeg = eeg.cpu().numpy().T
            eeg = self.resizeEEGToImageSize(eeg)
            eeg = torch.from_numpy(eeg)
        else:
            eeg = eeg.t()
        
        if self.data_augment_eeg:
            channel_norm_eeg = eeg
            for idx_channel in range(32):
                channel_index = np.random.randint(0, channel_norm_eeg.size(-1))
                channel_norm_eeg = self.normlizeEEG(channel_norm_eeg,ch_index=channel_index, class_index=None)

            z2Scoring = eeg
            fmean = z2Scoring.mean()
            fstd = z2Scoring.std()
            z2Scoring = (z2Scoring - fmean)/fstd
            
            # eeg_fft = torch.from_numpy(np.fft.fft(eeg.cpu().numpy()))
            eeg = torch.stack((eeg,channel_norm_eeg,z2Scoring))

        if self.add_channel_dim_to_eeg:
            eeg = eeg.unsqueeze(0)

        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

        label = self.class_labels_names[class_folder_name]

        if not self.inference_mode:
            label = label["ClassId"]
            if self.onehotencode_label:
                onehot = [0 for i in range(len(self.class_labels_names))]
                onehot[label] = 1
                label = onehot
                label = np.array(label)
                label = torch.from_numpy(label)

        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            image = self.preprocessin_fn(image, eeg, i, self, local_crops_to_remove=2)
        else:
            if self.perform_dinov2_global_Transform:
                image = self.geometric_augmentation_global(image)
            if self.convert_image_to_tensor:
                image = self.transform(image)

        if self.image_features_extracted==True and len(self.image_features)==len(self.images):
            image_features = self.image_features[i]
        else:
            image_features = []

        return eeg, label,image,i, image_features