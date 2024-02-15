import os
import time
import torch
import argparse
import numpy as np

from utils.EEGDataset import EEGDataset
from utils.Caltech101Dataset import Caltech101Dataset
from utils.DomainnetDataset import DomainnetDataset
from utils.CustomModel import CustomModel
from utils.ResnetFeatureRegressor import ResnetFeatureRegressor
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights


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
                        default='logs/ResnetfineTuned',
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
                        default="./logs/ResnetfineTuned/Resnet_finetuned_eeg__subject1_Final.pth",
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

    time_t1 = time.perf_counter()



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    weights = ResNet50_Weights.DEFAULT
    resnet50_model = resnet50(weights=weights)

    # Remove the final classification (softmax) layer
    model = torch.nn.Sequential(*(list(resnet50_model.children())[:-1])) 
    model.eval()
    model = model.to(device)

    # LSTM = LSTMModel(input_size=(LSTM_INPUT_FEATURES), hidden_size=(LSTM_HIDDEN_SIZE))
    # print(model)

    SUBJECT = FLAGS.subject
    BATCH_SIZE = int(FLAGS.batch_size)
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.dataset

    LSTM_INPUT_FEATURES = 2048 # should be image features output.
    LSTM_HIDDEN_SIZE = 460  # should be same as sequence length

    Use_ResnetFeatureRegressor = True


    # Load dataset
    dataset = EEGDataset(subset=FLAGS.search_gallary,eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())
    
    if FLAGS.query_gallary=="caltech101" :
        test_dataset = Caltech101Dataset(filter_label=FLAGS.class_to_search,images_path="./data/images/caltech/101_ObjectCategories",preprocessin_fn=weights.transforms())
    elif FLAGS.query_gallary=="domainnet" :
        test_dataset = DomainnetDataset(filter_label=FLAGS.class_to_search,images_path=f"./data/images/domainnet/{FLAGS.domainnet_subtype}",preprocessin_fn=weights.transforms())
    else:
        test_dataset = EEGDataset(subset=FLAGS.query_gallary,eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())
    # test_dataset = EEGDataset(subset=FLAGS.query_gallary,eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
    )

    # test_dataloader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=BATCH_SIZE,
    #         num_workers=0,
    #         shuffle=False,
    # )

    CustModel = CustomModel(input_size=(LSTM_INPUT_FEATURES),output_size=(LSTM_HIDDEN_SIZE*128))

    if os.path.exists(FLAGS.custom_model_weights):
        CustModel = torch.load(FLAGS.custom_model_weights)
        print(f"loaded custom weights: {FLAGS.custom_model_weights}")

    CustModel.to(device)


    topK  = FLAGS.topK

    Domainnet_toImagenet_Labels = { 

                                #    "airplane": "airliner",
                                #    "banana": "banana",
                                #    "cup": "coffee_mug",
                                #     "fish": "anemone_fish",
                                    # "piano": "grand_piano",
                                    # "butterfly": "lycaenid",
                                    # "camera": "reflex_camera",
                                    # "chair": "folding_chair",
                                    # "cell_phone": "cellular_telephone",
                                    # "dog":"German_shepherd",
                                    "guitar":"electric_guitar",
                                    # "elephant":"African_elephant",                        
                                    # "panda": "giant_panda",
                                    # "pizza": "pizza",
                                    # "violin": "electric_guitar",
                                    # "cat" : "wild_cat",
                                    # "shoe": "running_shoe",
                      

                                #  "laptop": "desktop_computer", 
                                #  "helicopter": "airliner",
                                #  "mandolin": "electric_guitar",
                                #  "inline_skate": "running_shoe",
                                #  "Leopards": "Egyptian_cat",
                                #  "cougar_face": "Egyptian_cat",
                                #  "cougar_body": "Egyptian_cat",
                                #  "dalmatian": "German_shepherd",
                                #  "crayfish": "anemone_fish"
                                # "butterfly":"lycaenid"
                                # 'water_lilly': 'daisy'
                                # "sunflower": "daisy"
                                
                                # 'strawberry': 'banana',
                                # 'umbrella': 'radio_telescope',
                                # 'starfish': 'anemone_fish',
                                # 'airplanes': 'missile',
                                # 'soccer_ball': 'golf_ball',

                                # 'schooner':'canoe',
                                # 'ketch': 'canoe',
                                # 'llama': 'sorrel',
                                # 'rhino': 'African_elephant',
                                # "sunflower": "daisy"


                                }
    


    for Domainnet_lebel, imagenetLabel in Domainnet_toImagenet_Labels.items():

        FLAGS.class_to_search  = Domainnet_lebel
        FLAGS.imagenet_label_name = imagenetLabel

        class_scores = {"data" :{}, "metadata": {}}
        class_scores["metadata"] = {"flags": FLAGS}

        test_dataset = DomainnetDataset(filter_label=FLAGS.class_to_search,images_path=f"./data/images/domainnet/{FLAGS.domainnet_subtype}",preprocessin_fn=weights.transforms())

        time_t0 = time.perf_counter()
        for tIdx, tesdD in enumerate(test_dataset):

            cosine_similarities = []
            cosine_similarities_labels_int = []
            cosine_similarities_labels_str = []
            cosine_similarities_labels_classid = []
            cosine_similarities_images = []

            test_eeg, test_label, test_image, test_idx = tesdD
            originalImage = test_dataset.getImagePath(test_idx)

            # if len(FLAGS.imagenet_label_name)>0 and FLAGS.imagenet_label_name!="":
            test_label["ClassName"] = imagenetLabel

            if test_label["ClassName"] not in class_scores["data"]:
                class_scores["data"][test_label["ClassName"]] = {"TP": 0, 
                                                        "classIntanceRetrival": 0,
                                                        "TotalRetrival": 0,
                                                        "TotalClass": 0, 
                                                        "input_images": [],
                                                        "GroundTruths": [], 
                                                        "Predicted":[], 
                                                        "Topk": {
                                                            "labels": [], 
                                                            "scores": [],
                                                            "images": []
                                                            }
                                                        }
            
            with torch.no_grad():
                if Use_ResnetFeatureRegressor:
                    outputs = CustModel(test_image.unsqueeze(0).to(device))
                else:
                    features = model(test_image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                    outputs = CustModel(features)

            test_eeg = outputs.cpu().numpy()

            time_t1 = time.perf_counter()
            for t_idx, train_data in enumerate(dataloader):
                eeg, label, image, idxs = train_data

                # print(eeg.shape)
                eeg_flattened = eeg.reshape(-1, eeg.size(1)*eeg.size(2))
                # eeg_flattened = eeg.reshape(dataloader.batch_size, -1)

                # Replicate feature vector to match batch size using broadcasting
                test_sample_batched = np.tile(test_eeg[np.newaxis, :, :], (eeg_flattened.shape[0], 1, 1))
                test_sample_batched = test_sample_batched.reshape(eeg_flattened.shape[0], -1)


                eeg_flattened = eeg_flattened.cpu().numpy()
                # print(eeg_flattened.shape)


                # print(test_sample_batched.shape, eeg_flattened.shape)
                similarities = cosine_similarity(test_sample_batched, eeg_flattened)
                # print(similarities[0])

                
                for cosSIm in abs(similarities[0].flatten()):
                    cosine_similarities.append(cosSIm)
                for intlabel in label["ClassId"].cpu().numpy().flatten():
                    cosine_similarities_labels_int.append(intlabel)
                for x in label["ClassName"]:
                    cosine_similarities_labels_str.append(x)

                for ClassId in label["ClassId"].cpu().numpy().flatten():
                    cosine_similarities_labels_classid.append(ClassId)


                idxs = idxs.cpu().numpy()
                for idx in idxs:
                    cosine_similarities_images.append(dataset.getImagePath(idx))

                # if t_idx>2:
                #     break
                    
            time_t2 = time.perf_counter()


            cosine_similarities = np.array(cosine_similarities, dtype=object)
            # print("cosine_similarities:",cosine_similarities.shape)
            cosine_similarities_labels_int = np.array(cosine_similarities_labels_int, dtype=object)
            # print("cosine_similarities_labels_int:",cosine_similarities_labels_int.shape)
            cosine_similarities_labels_str = np.array(cosine_similarities_labels_str, dtype=object)
            # print("cosine_similarities_labels_str:",cosine_similarities_labels_str.shape)
            cosine_similarities_labels_classid = np.array(cosine_similarities_labels_classid, dtype=object)
            # print("cosine_similarities_labels_classid:",cosine_similarities_labels_classid.shape)

            idx = np.argpartition(cosine_similarities, -topK)[-topK:]
            idx_top1 = np.argpartition(cosine_similarities, -1)[-1:]
            # print(idx)


            cosine_similarities_topk = cosine_similarities[idx]
            cosine_similarities_top1 = cosine_similarities[idx_top1]

            cosine_similarities_labels_classid_topk = cosine_similarities_labels_classid[idx]
            cosine_similarities_labels_classid_top1 = cosine_similarities_labels_classid[idx_top1]

            cosine_similarities_labels_int_topk = cosine_similarities_labels_int[idx]
            cosine_similarities_labels_int_top1 = cosine_similarities_labels_int[idx_top1]

            cosine_similarities_labels_str_topk = cosine_similarities_labels_str[idx]
            cosine_similarities_labels_str_top1 = cosine_similarities_labels_str[idx_top1]

            cosine_similarities_images_topk =  [cosine_similarities_images[i] for i in idx]
            cosine_similarities_images_top1 =  [cosine_similarities_images[i] for i in idx_top1]

            
            unique, counts = np.unique(cosine_similarities_labels_str_topk, return_counts=True)
            count = 0
            count_label = ""
            for u, c in zip(unique, counts):
                if u==test_label["ClassName"]:
                    count = c
                    count_label = u
            
            classIntanceRetrival = count
            TotalRetrival = topK

            if test_label["ClassName"] in cosine_similarities_labels_str_topk:
                class_scores["data"][test_label["ClassName"]]["TP"] +=1
                class_scores["data"][test_label["ClassName"]]["classIntanceRetrival"] +=classIntanceRetrival
                class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_label["ClassId"])
            else:
                class_scores["data"][test_label["ClassName"]]["Predicted"].append(dataloader.dataset.class_str_to_id[cosine_similarities_labels_str_top1[0]])

                
            class_scores["data"][test_label["ClassName"]]["TotalRetrival"] +=TotalRetrival
            class_scores["data"][test_label["ClassName"]]["TotalClass"] +=1

            class_scores["data"][test_label["ClassName"]]["Topk"]["labels"].append(list(cosine_similarities_labels_str_topk))
            class_scores["data"][test_label["ClassName"]]["Topk"]["scores"].append(list(cosine_similarities_topk))
            class_scores["data"][test_label["ClassName"]]["Topk"]["images"].append(list(cosine_similarities_images_topk))
            
            class_scores["data"][test_label["ClassName"]]["input_images"].append(originalImage)
            class_scores["data"][test_label["ClassName"]]["GroundTruths"].append(test_label["ClassId"])

            print(test_label, cosine_similarities_labels_str_topk,cosine_similarities_topk,cosine_similarities_labels_int_topk, f"Searched in: {time_t2-time_t1:.2f}s [{tIdx}/{len(test_dataset)}]")
        

        time_tn = time.perf_counter()

        output_dir = f"{FLAGS.log_dir}/{FLAGS.subject}"
        os.makedirs(output_dir,exist_ok=True)
        outputPath = f"{output_dir}/processed_data_Q_{FLAGS.query_gallary}_{FLAGS.domainnet_subtype}_S_{FLAGS.search_gallary}_{FLAGS.class_to_search}.pth"
        class_scores["metadata"] = {"processing_time": f"{time_tn-time_t0:.2f}s"}
        torch.save(class_scores, outputPath)
        print(f"Completed in : {time_tn-time_t0:.2f}")