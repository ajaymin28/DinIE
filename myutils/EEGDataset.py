import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
class EEGDataset(Dataset):
    # Constructor
    def __init__(self, eeg_signals_path, 
                 eeg_splits_path, subset='train',
                 subject=1, 
                 exclude_subjects=[],
                 filter_channels=[],
                 time_low=20,time_high=480, 
                 model_type="cnn", 
                 imagesRoot="./data/images/imageNet_images",
                 apply_norm_with_stds_and_means=False,
                 preprocessin_fn=None,
                 Transform_EEG2Image_Shape=False,
                 convert_image_to_tensor=False,
                 perform_dinov2_global_Transform=False,
                 dinov2Config=None,
                 linear_training_mode=False):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'


        self.Transform_EEG2Image_Shape = Transform_EEG2Image_Shape
        self.convert_image_to_tensor = convert_image_to_tensor
        self.perform_dinov2_global_Transform = perform_dinov2_global_Transform
        self.dinov2Config = dinov2Config
        self.apply_norm_with_stds_and_means = apply_norm_with_stds_and_means
        self.filter_channels = filter_channels
        self.linear_training_mode = linear_training_mode

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
        self.image_features = []

        self.class_labels = loaded["labels"]
        image_names = loaded['images']
        
        try:
            self.eeg_stds  = loaded["stddevs"]
            self.eeg_means  = loaded["means"]
        except Exception as e:
            print("error loading mean and std values")

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

        if subject==0:
            for sub in exclude_subjects:
                print(f"Subject: {sub} data will be exluded.")
        
        for sub_idx in subset_indexes:
            if self.apply_norm_with_stds_and_means:
                loaded['dataset'][sub_idx]["eeg"] = (loaded['dataset'][sub_idx]["eeg"]-self.eeg_means)/self.eeg_stds
            
            if subject!=0:
                if loaded['dataset'][sub_idx]['subject']==subject:
                    self.subsetData.append(loaded['dataset'][sub_idx])
                    self.labels.append(loaded["dataset"][sub_idx]['label'])
                    self.images.append(image_names[loaded["dataset"][sub_idx]['image']])
            else:
                data_subject = loaded['dataset'][sub_idx]['subject']
                if data_subject not in exclude_subjects:
                    sub_idx = int(sub_idx)
                    self.subsetData.append(loaded['dataset'][sub_idx])
                    self.labels.append(loaded["dataset"][sub_idx]['label'])
                    self.images.append(image_names[loaded["dataset"][sub_idx]['image']])

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


    def __len__(self):
        return self.size
        
    def generate_salt_pepper_noise(self, input_shape=(128,500)):
        im = np.zeros(input_shape, np.uint8) 
        noise = cv2.randu(im,(0),(255))
        return noise
    
    def transformToSaltPepperNoisyData(self):
        print("Transforming EEG data to salt_pepper_noise data")
        for i, image_path in enumerate(self.images):
            salt_pepper_noisy_data = self.generate_salt_pepper_noise()
            self.subsetData[i]["eeg"] = torch.from_numpy(salt_pepper_noisy_data).float()
        self.isDataTransformed = True
        print("Transforming EEG data to salt_pepper_noise (done)")

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
        
        #if not self.isDataTransformed:
        #    input_data = input_data[self.time_low:self.time_high,:]
        #print("input_data", input_data.shape)


        # print(input_data.shape)
        IMG_H, IMG_W, IMG_C  = imageShape[0], imageShape[1], imageShape[2]
        # EEG input_data is assumed to be a numpy array of shape (128, 460)
        
        #print("img", IMG_H, IMG_W, IMG_C )

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
        
    
    def __getitem__(self, i):
        eeg = self.subsetData[i]["eeg"].float()
        
        #print(eeg.size(), "original")
        
        if len(self.filter_channels)>0:
            eeg = eeg.cpu().numpy().T
            eeg_zeros = np.zeros((self.time_high-self.time_low, len(self.filter_channels)), dtype=eeg.dtype)
            #print(eeg_zeros.shape, "eeg_zeros", "original eeg", eeg.shape)
            # print(f"eeg shape: {eeg.shape} eeg zeros shape: {eeg_zeros.shape}")
            for ch_idx, ch in enumerate(self.filter_channels):
                #print(f" sliced: {eeg[self.time_low:self.time_high,ch].shape}")
                eeg_zeros[:,ch_idx] = eeg[self.time_low:self.time_high,ch]
            eeg = torch.from_numpy(eeg_zeros).t()
            
            # print(f"final EEG : {eeg.size()}")
        #else:
        #    eeg = eeg[self.time_low:self.time_high,:]
        #print(eeg.size(), "after")
        
        
        
        if self.Transform_EEG2Image_Shape:
            eeg = eeg.cpu().numpy().T
            #print("resziging", eeg.shape)
            eeg = self.resizeEEGToImageSize(eeg)
            eeg = torch.from_numpy(eeg)
        else:
            eeg = eeg.t()

        #if not self.isDataTransformed:
        #    if len(self.filter_channels)>0:
        #        eeg = eeg.cpu().numpy().T
        #        eeg_zeros = np.zeros((self.time_high-self.time_low, len(self.filter_channels)), dtype=eeg.dtype)
        #        # print(f"eeg shape: {eeg.shape} eeg zeros shape: {eeg_zeros.shape}")
        #        for ch_idx, ch in enumerate(self.filter_channels):
        #            # print(f" sliced: {eeg[self.time_low:self.time_high,ch].shape}")
        #            eeg_zeros[:,ch_idx] = eeg[self.time_low:self.time_high,ch]
        #        eeg = torch.from_numpy(eeg_zeros).t()
        #        # print(f"final EEG : {eeg.size()}")
        #    else:
        #        eeg = eeg[self.time_low:self.time_high,:]

        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

        label = self.class_labels_names[class_folder_name]

        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            if self.linear_training_mode:
              image = self.preprocessin_fn(image)
            else:
              # image = self.preprocessin_fn(image, eeg, i, self, local_crops_to_remove=2)
              image = self.preprocessin_fn(image, eeg, i, self, local_crops_to_remove=2)
        else:
            if self.perform_dinov2_global_Transform:
                image = self.geometric_augmentation_global(image)
            if self.convert_image_to_tensor:
                image = self.transform(image)
                # print(image.size())

        if self.image_features_extracted==True and len(self.image_features)==len(self.images):
            image_features = self.image_features[i]
        else:
            image_features = []

        return eeg, label["ClassId"],image,i, image_features