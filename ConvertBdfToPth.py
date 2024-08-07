from mne.io import read_raw_bdf
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from sklearn.manifold import TSNE
import pickle
import torch
import os

"""
Spampinato Dataset Struct

data = {
    "dataset": [
        {
            'eeg': torch.tensor([]),
            'image': 0,  # image index from data["images"] 
            'label': 10,
            'subject': 4
        }
    ],
    "labels": [], # 40 imagenet classes list e.g. n03709823
    "images": [], # image names e.g. n03709823_25676
    "means": [], # means torch tensor 1,128
    "stddevs": [] # stddev torch tensor 1,128
}

We replicate this for The Perils Dataset
"""

class DESIGN:
    BLOCK = 1
    VIDEO = 2
    RAPID_EVENT = 3
    VIDEO_RAPID = 4


class FLAGS:
    SUBJECT = 2
    EEG_DESIGN = DESIGN.RAPID_EVENT
    BLOCK_NUMBER = 1 # only applicable when BLOCK desing is used.
    Apply_high_low_pass_filter = False
    LOW_PASS_CUT_FREQ = 5
    HIGH_PASS_CUT_FREQ = 96
    Replace_Rapid_Image_Sequence_With_Block_Sequence = True
    take_alternate_DownSamples = False  # if provided true every DownRatio_for_timesamples'th will be taken from 2048 time samples else fixed first 512 samples will be taken
    
    """
    Don't change any FLAGS below, unless you know what you are doing.
    """

    EEG_reference_channels = [96,97] # "EXG1", "EXG2", since in python index starts with 0
    Apply_notch_filter = False # since re referrencing removes the line noise as well so no need to apply notch filter on this. 
    Notch_Filter_Freq = 60  # Hz for USA
    EEG_DATA_TYPE = "IMAGE_BLOCK"
    DATA_PATH = "./data"
    OUT_DIR = "./output"
    FilePath  = ""
    image_sequence = ""
    index_ = 1

    Number_of_image_samples = 2000
    DownSampling_Frequency_ratio = 0.5
    DownRatio_for_timesamples = 4
    BAD_CHANNELS = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7",  "EXG8", "Status"] # will be dropped in final stage before saving
    STATUS_EVENT  = 65280

    image_class_mappings_file = "./data/image.pkl"
    image_block_event_sequence = "./data/image-block.txt"
    image_rapid_event_sequence = "./data/image-rapid-event.txt"


if __name__=="__main__":
    """
    Indexes are taken from the matlab code provided in the Perils Dataset
    """
    FLAGS.index_ = 1
    FLAGS.image_sequence = FLAGS.image_block_event_sequence
    if FLAGS.EEG_DESIGN==DESIGN.RAPID_EVENT:
        FLAGS.index_ = DESIGN.RAPID_EVENT
        if FLAGS.SUBJECT==6:
            FLAGS.index_ = 7
        if FLAGS.SUBJECT>1 and FLAGS.SUBJECT<6:
            FLAGS.index_ = 5
        FLAGS.image_sequence = FLAGS.image_rapid_event_sequence 
        if FLAGS.Replace_Rapid_Image_Sequence_With_Block_Sequence:
            FLAGS.image_sequence = FLAGS.image_block_event_sequence # experimental
        FLAGS.EEG_DATA_TYPE = "IMAGE_RAPID"
    elif FLAGS.EEG_DESIGN==DESIGN.BLOCK:
        FLAGS.index_ = DESIGN.BLOCK
        if FLAGS.BLOCK_NUMBER==2:
            FLAGS.index_ = FLAGS.BLOCK_NUMBER
        FLAGS.image_sequence = FLAGS.image_block_event_sequence
        FLAGS.EEG_DATA_TYPE = "IMAGE_BLOCK"
    elif FLAGS.EEG_DESIGN==DESIGN.VIDEO:
        FLAGS.index_ = DESIGN.VIDEO
        FLAGS.EEG_DATA_TYPE = "VIDEO"
    elif FLAGS.EEG_DESIGN==DESIGN.VIDEO_RAPID:
        FLAGS.index_ = DESIGN.VIDEO_RAPID
        FLAGS.EEG_DATA_TYPE = "VIDEO_RAPID"

    FLAGS.FilePath = f"{FLAGS.DATA_PATH}/spampinato-{FLAGS.SUBJECT}-{FLAGS.index_}.bdf"
    if not os.path.exists(FLAGS.FilePath):
        print(f"File {FLAGS.FilePath} doesnt exist.")
    
    raw = read_raw_bdf(FLAGS.FilePath, preload=True)
    channel_names = raw.ch_names
    print(channel_names)
    print (f"channel 97: {channel_names[FLAGS.EEG_reference_channels[0]]} 98: {channel_names[FLAGS.EEG_reference_channels[1]]}") # since in python index starts with 0
    
    # this re referrencing removes the line noise as well so no need to apply notch filter on this. 
    raw = raw.set_eeg_reference(ref_channels=[channel_names[FLAGS.EEG_reference_channels[0]],channel_names[FLAGS.EEG_reference_channels[1]]])  # setting channel 97 and 98 as reference channels
    # raw.plot()

    NumberOfSamples_n = FLAGS.Number_of_image_samples
    Sampling = int(raw.info["sfreq"]*FLAGS.DownSampling_Frequency_ratio) # 4096==> 2048
    TimeSamplesToTake = Sampling//FLAGS.DownRatio_for_timesamples # 2048/4 ==> 512
    # raw = raw.resample(sfreq=Sampling)

    # Modify Events location
    """
    MNE expects the actual event in the last axis.
    """
    default_events = mne.find_events(raw, initial_event=False, stim_channel="Status")
    default_events = default_events[1:]  # remove first event, its useless.
    modified_events = default_events.copy()
    modified_events[:,0] = default_events[:,0]
    modified_events[:,1] = default_events[:,2]
    modified_events[:,2] = default_events[:,1]
    raw.add_events(events=modified_events,stim_channel="Status", replace=True)
    events = mne.find_events(raw, initial_event=False, stim_channel="Status")
    events.shape
    assert events[0][-1]==FLAGS.STATUS_EVENT  # check if event is in last axis.

    TimeBetweenTwoEvents = events[2][0]-events[1][0]
    print("Time between two events",TimeBetweenTwoEvents)

    print("First 10 Events interval: ")
    for i in range(10):
        print(f"[{events[i][0]/Sampling:.2f}s]" ,end =" ")

    
    if FLAGS.Apply_high_low_pass_filter:
        raw = raw.filter(l_freq=FLAGS.LOW_PASS_CUT_FREQ, h_freq=FLAGS.HIGH_PASS_CUT_FREQ)
    
    if FLAGS.Apply_notch_filter:
        freqs = [FLAGS.Notch_Filter_Freq]  # Frequencies to be notched out (example: 50Hz for power-line noise, 60Hz for power-line noise USA)
        raw = raw.notch_filter(freqs, filter_length=13518, phase='zero') #13518 reffered from matlab code provided by The perils and pitfalls of eeg paper

    
    # Drop unneccessary channels 
    Channels_before = len(channel_names)
    raw.info["bads"] = FLAGS.BAD_CHANNELS
    RemainingChannels = Channels_before - len(FLAGS.BAD_CHANNELS)
    raw = raw.drop_channels(FLAGS.BAD_CHANNELS)
    print(f"RemainingChannels: {RemainingChannels}")

    np_raw = raw.get_data()
    print("np_raw ",np_raw.shape)

    EEG = np.zeros((NumberOfSamples_n,TimeSamplesToTake,RemainingChannels))
    # print(EEG.shape)
    SamplesAdded = 0

    if FLAGS.take_alternate_DownSamples:
        for i, event in enumerate(events): # ignore first event here.
            if event[-1]==FLAGS.STATUS_EVENT:
                next_event_index = -1
                if i!=len(events):
                    next_event_index = i + 1
                if next_event_index==-1:
                    current_sample_data_till_next_event = np_raw[:, event[0]:]
                else:
                    current_sample_data_till_next_event = np_raw[:, event[0]:event[0]+Sampling]
                sample_data = current_sample_data_till_next_event[:, ::FLAGS.DownRatio_for_timesamples] # Take every 4th sample
                EEG[i,:,:] =  sample_data.T
                SamplesAdded+=1
    else:
        for i, event in enumerate(events): # ignore first event here.
            if event[-1]==FLAGS.STATUS_EVENT:
                sample_data = np_raw[:, event[0]:event[0]+TimeSamplesToTake]
                # print(i ,sample_data.shape, event[0],event[0]+TimeSamplesToTake)
                EEG[i,:,:] =  sample_data.T
                SamplesAdded+=1
    # print("EEG Shape", EEG.shape)
    TimeSamples = EEG.shape[1]
    EEG_reshaped = EEG.reshape(-1, RemainingChannels)
    print(f"Samples added: {SamplesAdded}")
    assert NumberOfSamples_n==SamplesAdded


    OVERALL_EEG_MEAN = np.mean(EEG_reshaped,axis=0)
    OVERALL_EEG_STD = np.std(EEG_reshaped,axis=0)
    # OVERALL_EEG_STD.shape

    # for ch_idx in range(RemainingChannels):
    #     EEG_reshaped[:, ch_idx] = (EEG_reshaped[:, ch_idx]-OVERALL_EEG_MEAN[ch_idx])/OVERALL_EEG_STD[ch_idx]
    """
    Norm is not applied, since while loading the dataset we have option to apply the norm or skip it.
    """


    EEG = EEG_reshaped.reshape(NumberOfSamples_n,TimeSamples,RemainingChannels)

    file = open(FLAGS.image_class_mappings_file,'rb')
    image_class_mappings = pickle.load(file)
    file.close()

    imagenet_class_to_int_mapping = {}
    for key,val in image_class_mappings.items():
        key_slice = key.split("_")[0]
        if key_slice not in imagenet_class_to_int_mapping:
            imagenet_class_to_int_mapping[key_slice] = val

    images_list = []
    images_classes = []
    image_names_without_extension = []
    with open(FLAGS.image_sequence) as f:
        images_list = f.readlines()
        for i,line in enumerate(images_list):
            line = line.strip()
            imageName = line.split(".")[0]
            images_classes.append(image_class_mappings[imageName])
            image_names_without_extension.append(imageName)


    Data = {
        "dataset": [],
        "labels": [], # 40 imagenet classes list e.g. n03709823
        "images": [], # image names e.g. n03709823_25676
        "means": [], # means torch tensor 1,128
        "stddevs": [], # stddev torch tensor 1,128
        "config": []
    }

    Data["images"]  = image_names_without_extension
    Data["labels"]  = list(imagenet_class_to_int_mapping.keys())
    Data["means"].append(torch.from_numpy(OVERALL_EEG_MEAN))
    Data["stddevs"].append(torch.from_numpy(OVERALL_EEG_STD))
    
    Config = {key:value for key, value in FLAGS.__dict__.items() if not key.startswith('__') and not callable(key)}
    # Design_Config = {key:value for key, value in DESIGN.__dict__.items() if not key.startswith('__') and not callable(key)}
    # Config = dict(list(Config.items()) + list(Design_Config.items()))
    Data["config"].append(Config)
    print(Config)


    for i in range(NumberOfSamples_n):
        EEG_DATA = {
            'eeg': torch.from_numpy(EEG[i,:,:].T),  # channel first
            'image': i,  # image index from data["images"] 
            'label': images_classes[i],
            'subject': FLAGS.SUBJECT
        }

        Data["dataset"].append(EEG_DATA)

    os.makedirs(FLAGS.OUT_DIR, exist_ok=True)
    FILE_NAME = f"spampinato-{FLAGS.SUBJECT}-{FLAGS.EEG_DATA_TYPE}"
    if FLAGS.Apply_high_low_pass_filter:
        FILE_NAME += f"_{FLAGS.LOW_PASS_CUT_FREQ}Hz_{FLAGS.HIGH_PASS_CUT_FREQ}Hz"
    else:
        FILE_NAME += f"_RAW_with_mean_std"
        
    
    if FLAGS.Replace_Rapid_Image_Sequence_With_Block_Sequence:
        torch.save(Data, f"{FLAGS.OUT_DIR}/{FILE_NAME}_experimental_given_blockimageSeqToRapid.pth")
    else:
        torch.save(Data, f"{FLAGS.OUT_DIR}/{FILE_NAME}.pth")