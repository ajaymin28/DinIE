import torch
import argparse
import os
import torch.nn as nn
import uuid
import faiss
import json
from utils.PerilsEEGDataset import EEGDataset
import time
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from utils import utils


class HyperParams:
    learning_rate=0.001
    T=0.5
    soft_target_loss_weight=0.25
    ce_loss_weight=0.75
    warmup_teacher_temp = 1.65
    teacher_temp = 0.22
    warmup_teacher_temp_epochs = 50


class Parameters:
    ce_loss_weight = 0.95
    mse_loss_weight = 0.20
    soft_target_loss_weight = 0.05
    alpha = 0
    temperature = 0.33
    teacher_temp = 0.33
    student_temp= 0.24

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, student_outputs, teacher_outputs):
        loss = 1 - self.cosine_similarity(student_outputs, teacher_outputs).mean()
        return loss

def cosine_similarity_loss(v1, v2):
    """
    Compute the cosine similarity loss between two feature vectors.

    Args:
        v1 (torch.Tensor): First feature vector (shape: [batch_size, feature_dim]).
        v2 (torch.Tensor): Second feature vector (shape: [batch_size, feature_dim]).

    Returns:
        torch.Tensor: Cosine similarity loss.
    """
    # Normalize the vectors
    v1_normalized = F.normalize(v1, p=2, dim=1)
    v2_normalized = F.normalize(v2, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = torch.sum(v1_normalized * v2_normalized, dim=1)

    # Define the loss as negative cosine similarity
    loss = -torch.mean(cosine_sim)

    return loss


class FeatureDistributionLoss(nn.Module):
    def __init__(self, nepochs, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs):
        super().__init__()
        self.mse = nn.MSELoss()

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_outputs, teacher_outputs, epoch, label, pred_label=None):
        overall_loss = 0.0

        # distribution loss
        # student_mean, student_std = student_outputs.mean(), student_outputs.std()
        # teacher_mean, teacher_std = teacher_outputs.mean(), teacher_outputs.std()
        # mean_mse = self.mse(student_mean, teacher_mean)
        # mean_std = self.mse(student_std, teacher_std)
        # overall_loss += mean_mse
        # overall_loss += mean_std

        HyperParams.T = self.teacher_temp_schedule[epoch]

        # # # mse_loss = self.mse(student_outputs, teacher_outputs)
        # student_out = student_outputs / HyperParams.T
        # teacher_out = F.softmax(teacher_outputs/HyperParams.T, dim=-1)
        # # teacher_out = teacher_out.detach().chunk(2)
        # loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        # overall_loss += loss.mean()

        # ce_loss = nn.functional.cross_entropy(nn.functional.softmax(pred_label,dim=-1), label.float())
        
        # overall_loss += ce_loss

        # ce_loss_logits = nn.functional.cross_entropy(student_outputs, teacher_outputs) # is negative 

        # soft_targets = nn.functional.softmax(teacher_outputs / HyperParams.T, dim=-1).to(device)
        # soft_prob = nn.functional.log_softmax(student_outputs / HyperParams.T, dim=-1).to(device)
        # # # softLabelLoss = nn.functional.cross_entropy(soft_prob, soft_targets)
        # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)
        # overall_loss += ce_loss_logits


        HyperParams.T = self.teacher_temp_schedule[epoch]
        soft_targets = nn.functional.softmax(teacher_outputs / HyperParams.T, dim=-1).to(device)
        soft_prob = nn.functional.log_softmax(student_outputs / HyperParams.T, dim=-1).to(device)
        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)
        # overall_loss += soft_targets_loss

        # mse_loss = self.mse(student_outputs, teacher_outputs)
        # overall_loss = HyperParams.soft_target_loss_weight * soft_targets_loss + HyperParams.ce_loss_weight * ce_loss

        overall_loss =  soft_targets_loss
        return overall_loss


def loss_fn_kd(student_logits, teacher_logits,gt_labels, pred_labels, params:Parameters):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    loss = 0.0
    alpha = params.alpha
    T = params.temperature
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                          F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
    #           F.cross_entropy(outputs, labels) * (1. - alpha)

    #Soften the student logits by applying softmax first and log() second
    # soft_targets = nn.functional.softmax(teacher_logits, dim=-1).to(device)
    # soft_prob = nn.functional.log_softmax(student_logits, dim=-1).to(device)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

    # gt_labels = gt_labels["ClassId"]
    # print(gt_labels.size(), pred_labels.size())

    # gt_labels = gt_labels.view(gt_labels.size(0), 1).float()
    # print(pred_labels.size(), gt_labels.size())
    ce_loss = F.cross_entropy(nn.functional.softmax(pred_labels.float(),dim=-1), gt_labels.float())

    # cosine_loss = cosine_similarity_loss(student_logits, teacher_logits)

    # mse_loss = F.mse_loss(student_logits, teacher_logits) * params.mse_loss_weight
                        
    # Weighted sum of the two losses
    # loss = params.soft_target_loss_weight * soft_targets_loss + params.ce_loss_weight * mse_loss
    # KD_loss = nn.KLDivLoss()(F.log_softmax(student_logits/T, dim=1), F.softmax(teacher_logits/T, dim=1) * (alpha * T * T) + ce_loss * (1. - alpha))


    # loss = ce_loss * params.ce_loss_weight + soft_targets_loss * params.soft_target_loss_weight
    # loss += soft_targets_loss
    loss += ce_loss

    return loss



class FLAGS:
    num_workers = 4
    dist_url = "env://"
    local_rank = 0
    batch_size = 4

def initDinoV2Model(model= "dinov2_vits14"):
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", model)
    return dinov2_vits14

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=4, out_features=384, number_of_classes=40):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)
        self.class_pred = nn.Linear(out_features, number_of_classes)
    
    def forward(self, x):
        batch_size, channels, timespan = x.size()
        # x = x.view(batch_size, channels, timespan)
        lstm_init = (torch.zeros(self.n_layer, batch_size, self.hidden_size), torch.zeros(self.n_layer, batch_size, self.hidden_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        # lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]

        # lstm_out, (ht, ct) = self.lstm(x)
        # print(lstm_out.size(), ht.size(), ct.size())

        # hx0, hx1 =  self.lstm(x, lstm_init)[1]
        # print(hx0.size(), hx1.size())
        # x = F.softmax(self.fc(x))
        x = self.fc(x)
        cls_pred = self.class_pred(x)
        x = nn.functional.relu(x)
        return x, cls_pred
        # h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # lstm_out, hidden_out = self.lstm(x, (h0, c0))
        # # out = self.fc(lstm_out[:, -1, :])
        # return lstm_out 


if __name__=="__main__":


    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./logs/FixedSamples/withnorm',
                        help='Directory to put logging.')
    parser.add_argument('--gallery_subject',
                        type=int,
                        default=1,
                        choices=[0,1,2,3,4,5,6],
                        help='Subject Data to train')
    parser.add_argument('--query_subject',
                        type=int,
                        default=1,
                        choices=[0,1,2,3,4,5,6],
                        help='Subject Data to train')
    parser.add_argument('--eeg_dataset',
                        type=str,
                        default="./data/eeg/theperils/FixedSampling/spampinato-1-IMAGE_RAPID_RAW_with_mean_std_experimental_given_blockimageSeqToRapid.pth",
                        help='Dataset to train')
    parser.add_argument('--images_root',
                        type=str,
                        default="./data/images/imageNet_images",
                        help='Dataset to train')
    parser.add_argument('--eeg_dataset_split',
                        type=str,
                        default="./data/eeg/block_splits_by_image_all.pth",
                        help='Dataset split')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--custom_model_weights',
                        type=str,
                        default="./models/dino/localcrops_as_eeg/subject1/checkpoint.pth",
                        help='custom model weights')
    parser.add_argument('--dino_base_model_weights',
                        type=str,
                        default="./models/pretrains/dino_deitsmall8_pretrain_full_checkpoint.pth",
                        help='dino based model weights')
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--query_gallery',
                        type=str,
                        default="test",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--gallery_tranformation_type',
                        type=str,
                        default="eeg2eeg",
                        choices=["img", "img2eeg", "eeg", "eeg2eeg"],
                        help='type of tansformation needed to be done to create search gallery')
    parser.add_argument('--query_tranformation_type',
                        type=str,
                        default="eeg2eeg",
                        choices=["img", "img2eeg", "eeg", "eeg2eeg"],
                        help='type of tansformation needed to be done to create query instances')
    parser.add_argument('--hyperprams',
                        type=str,
                        default="{'ce_loss_weight': 0.50, 'soft_target_loss_weight':0.50,'alpha': 1,'temperature':2}",
                        help='hyper params for training model, pass dict tpye in string format')
    parser.add_argument('--seed', default=43, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    SUBJECT = FLAGS.gallery_subject
    BATCH_SIZE = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.eeg_dataset

    FileName = EEG_DATASET_PATH.split("/")[-1]
    FileName = FileName.split(".")[0]

    FLAGS.log_dir = os.path.join(FLAGS.log_dir, FileName)
    os.makedirs(FLAGS.log_dir, exist_ok=True)

    # EEG_DATASET_SPLIT = "./data/eeg/block_splits_by_image_all.pth"

    # LSTM_INPUT_FEATURES = 128 # should be image features output.
    # LSTM_HIDDEN_SIZE = 460  # should be same as sequence length
    selectedDataset = "imagenet40"

    hyperprams = eval(FLAGS.hyperprams)
    if 'alpha' in hyperprams:
        Parameters.alpha = hyperprams["alpha"]
    if 'ce_loss_weight' in hyperprams:
        Parameters.ce_loss_weight = hyperprams["ce_loss_weight"]
    if 'soft_target_loss_weight' in hyperprams:
        Parameters.soft_target_loss_weight = hyperprams["soft_target_loss_weight"]
    if 'temperature' in hyperprams:
        Parameters.temperature = hyperprams["temperature"]

    utils.init_distributed_mode(FLAGS)

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),       
        transforms.CenterCrop(224),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    ])

    dataset = EEGDataset(subset="train",
                         eeg_signals_path=EEG_DATASET_PATH,
                         eeg_splits_path=None, 
                         subject=SUBJECT,
                         time_low=0,
                         imagesRoot=FLAGS.images_root,
                         time_high=480,
                         exclude_subjects=[],
                         convert_image_to_tensor=False,
                         apply_channel_wise_norm=False,
                         apply_norm_with_stds_and_means=True,
                         preprocessin_fn=transform_image,
                         inference_mode=False,
                         onehotencode_label=True)


    eeg, label,image,i, image_features =  dataset[0]

    temporal_length, channels = eeg.size()
    print(temporal_length,channels)

    LSTM_INPUT_FEATURES = channels
    LSTM_HIDDEN_SIZE = int(temporal_length/2)  # should be same as sequence length


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    dinov2_model = dinov2_model.eval()
    dinov2_model.to(device)

    data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    dataset.extract_features(model=dinov2_model, data_loader=data_loader_train, replace_eeg=False)

    generator1 = torch.Generator().manual_seed(43)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)
    data_loader_train = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_val = DataLoader(val_ds, batch_size=FLAGS.batch_size, shuffle=True)


    eeg, label,image,i, image_features = next(iter(data_loader_train)) 
    outs = dinov2_model(image.to(device))
    features_length = outs.size(-1)
    print("Dino Features size", outs.size(), "eeg size ", eeg.size())

    model = LSTMModel(input_size=LSTM_INPUT_FEATURES,hidden_size=LSTM_HIDDEN_SIZE, out_features=features_length, n_layers=4)
    model.to(device)

    output, cls_pred = model(eeg.to(device))
    print(output.size())


    opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    criterion_feature_dist = FeatureDistributionLoss(nepochs=EPOCHS, 
                                        warmup_teacher_temp=HyperParams.warmup_teacher_temp, 
                                        teacher_temp=HyperParams.teacher_temp,
                                        warmup_teacher_temp_epochs=HyperParams.warmup_teacher_temp_epochs)

    epoch_losses = []
    val_epoch_losses = []
    best_val_loss = None
    epochs_since_val_loss_improvement = 0
    early_stop_patience = 5
    best_loss_state_dict = None
    for EPOCH in range(EPOCHS):

        batch_losses = []
        val_batch_losses = []
        val_loss_improved = False

        model.train()

        for data in data_loader_train:
            eeg, label,image,i, image_features = data

            image_features = torch.from_numpy(np.array(image_features)).to(device)

            opt.zero_grad()
            lstm_output, cls_pred = model(eeg.to(device))
            loss = criterion_feature_dist(lstm_output, image_features, EPOCH, label.to(device), pred_label=cls_pred)
                        
            batch_losses.append(loss.cpu().item())

            loss.backward()
            opt.step()

        model.eval()

        for data in data_loader_val:
            eeg, label,image,i, image_features = data

            with torch.no_grad():
                image_features = torch.from_numpy(np.array(image_features)).to(device)
                lstm_output, cls_pred = model(eeg.to(device))
                loss = criterion_feature_dist(lstm_output, image_features, EPOCH, label.to(device), pred_label=cls_pred)
                loss = loss.cpu().item()
                val_batch_losses.append(loss)
        
        batch_losses = np.array(batch_losses)
        val_batch_losses = np.array(val_batch_losses)
        val_epoch_loss= val_batch_losses.mean()
        epoch_loss = batch_losses.mean()
        epoch_losses.append(epoch_loss)
        val_epoch_losses.append(val_epoch_loss)

        if best_val_loss is None:
            best_val_loss = val_epoch_loss
        else:
            if val_epoch_loss<best_val_loss:
                val_loss_improved = True
                best_val_loss = val_epoch_loss
                epochs_since_val_loss_improvement = 0
                best_loss_state_dict = model.state_dict()
                model_dict = {
                    "state_dict": best_loss_state_dict,
                    "losses": epoch_losses,
                    "val_losses": val_epoch_losses,
                }
                torch.save(model_dict, f"{FLAGS.log_dir}/lstm_dinov2_best_loss.pth")

        if not val_loss_improved:
            epochs_since_val_loss_improvement +=1

        print(f"EPOCH {EPOCH} train_loss: {round(epoch_loss,6)} val_loss: {round(val_epoch_loss,6)} Epoch since val loss improved: {epochs_since_val_loss_improvement}")

        if epochs_since_val_loss_improvement>early_stop_patience:
            print("Early Stopping since val loss is not improving.")
            break
    
    model_dict = {
        "state_dict": best_loss_state_dict,
        "losses": epoch_losses,
        "val_losses": val_epoch_losses,
    }
    torch.save(model_dict, f"lstm_dinov2_learned_features_final.pth")