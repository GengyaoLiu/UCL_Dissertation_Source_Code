import math
import os
import numpy as np
import torch
import logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter

import util
from model.VitPatch16 import VIT_patch16
from model.VitPatch32 import VIT_patch32
from model.ResNet34 import ResNet_34
from model.ResNet50 import ResNet_50
from model.ResVit import ResVit
from augmentation import augment
from args import get_train_args
from util import str2bool


def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    fileHandler = logging.FileHandler(log_file, mode='a')

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    return logger


def split_dataset(root_dir, csv_file='labels.csv', train_size=0.8, valid_size=0.1):
  train_dir = "/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/train_df.csv"
  valid_dir = "/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/valid_df.csv"
  test_dir = "/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/test_df.csv"
  if not os.path.exists(train_dir) and \
      not os.path.exists(valid_dir) and \
      not os.path.exists(test_dir):
      # Load labels from CSV
      df = pd.read_csv(os.path.join(root_dir, csv_file))

      # Encode labels to numerical data,
      # 0 for "happy", 1 for "sad", 2 for "angry", 3 for "relaxed"
      label_encoder = LabelEncoder()
      df['label'] = label_encoder.fit_transform(df['label'])

      # Split into train and temp (test + validation)
      train_data, temp_data = train_test_split(df, train_size=train_size, random_state=100, shuffle=True,
                                              stratify=df['label'])

      # Determine the test size such that the remaining data will be used for validation
      test_size = 1.0 - (valid_size / (1.0 - train_size))

      # Split temp_data into validation and test
      valid_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=100, shuffle=True,
                                              stratify=temp_data['label'])
      print("Data set is split into: \n Training: ", train_data.shape[0],
            "\n Validation: ", valid_data.shape[0],
            "\n Test: ", test_data.shape[0])
      train_data.to_csv(train_dir, index=False)
      valid_data.to_csv(valid_dir, index=False)
      test_data.to_csv(test_dir, index=False)
  else:
      train_data = pd.read_csv(train_dir)
      valid_data = pd.read_csv(valid_dir)
      test_data = pd.read_csv(test_dir)

  return train_data, valid_data, test_data


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(SupervisedContrastiveLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None):
        """
        input:
            features: input embeddings [batch_size, hidden_dim]
            labels: label of all samples [batch_size].
            mask: mask for learning [batch_size, batch_size], if sample i and j have the same label，then mask_{i,j}=1,
                  0 otherwise
        output:
            loss value
        """
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            raise ValueError('Label is required to compute contrastive loss')

        # compute logits
        # similarity of samples i j
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=mask.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)  # beside itself, num of positives  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class HybridLoss(nn.Module):
    def __init__(self, device, args, temperature=0.1, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(device, temperature)
        self.alpha = alpha
        self.args = args

    def forward(self, output, embeddings, labels):
        if self.args.loss_name == "Hybrid_loss":
          cross_entropy_loss = self.cross_entropy_loss(output, labels)
          contrastive_loss = self.contrastive_loss(embeddings, labels)
          return self.alpha * cross_entropy_loss + (1 - self.alpha) * contrastive_loss
        else:
          cross_entropy_loss = self.cross_entropy_loss(output, labels)
          return cross_entropy_loss


class DogExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, args, root_dir, df, dataset, transform=None):
        if not str2bool(args.augment):
          self.root_dir = root_dir
          self.transform = transform
          self.image_files = [os.path.join(self.root_dir, x) for x in df['filename']]
          self.image_labels = [x for x in df['label']]
        else:
          if dataset == "Train" or dataset == "Validate":
            self.root_dir = '/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/images_augment'
            self.transform = transform
            self.image_files = [os.path.join(self.root_dir, x) for x in df['filename']]
            self.image_labels = [x for x in df['label']]
          else:
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = [os.path.join(self.root_dir, x) for x in df['filename']]
            self.image_labels = [x for x in df['label']]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def train(device, epoch_num, logger, writer, dataloader, model, loss_fn, optimizer):
    model.train()
    batch = 0
    avgloss = 0
    with tqdm(dataloader, unit="batch") as tqdm_epoch:
        for X, y in tqdm_epoch:

            tqdm_epoch.set_description(f"Epoch:")
            # print(X)
            X, y = X.to(device), y.to(device)
            embeddings, logits = model(device, X)
            loss = loss_fn(output=logits, embeddings=embeddings, labels=y.long().squeeze())
            avgloss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1
            tqdm_epoch.set_postfix(loss=(avgloss / batch).item(), lr=optimizer.param_groups[0]['lr'])

            # Calculate metrics
            pred_argmax = logits.argmax(1).detach().cpu().numpy()
            y_true = y.long().squeeze().cpu().numpy()
            acc = accuracy_score(y_true, pred_argmax)
            f1 = f1_score(y_true, pred_argmax, average='weighted')

            # Log metrics to TensorBoard
            writer.add_scalar('Train/Loss', loss.item(), epoch_num)
            writer.add_scalar('Train/Accuracy', acc, epoch_num)
            writer.add_scalar('Train/F1-score', f1, epoch_num)

    print(f"Training Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {(avgloss / batch).item():>7f} \n")


def test(device, epoch_num, logger, writer, dataloader, model, loss_fn, scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    confusion_pred, confusion_label =[], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embeddings, logits = model(device, X)
            # sum up the test loss per batch
            test_loss += loss_fn(output=logits, embeddings=embeddings, labels=y.long().squeeze()).item()
            # sum up the number of correct predictions per batch
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
            # predicted labels [Type: A list of lists, list of batches of predictions]
            confusion_pred.append(list(logits.argmax(1).cpu().numpy()))
            # print("shape of logits: ", logits.shape)

            # actual labels [Type: A list of lists, list of batches of labels]
            confusion_label.append(list(y.cpu().numpy()))
    scheduler.step()
    # loss of each batch
    test_loss /= num_batches
    # accuracy rate
    correct /= size
    # flatten the list of lists into an array
    confusion_pred = np.array(sum(confusion_pred, []))
    # print("predictions: ",confusion_pred)
    # flatten the list of lists into an array
    confusion_label = np.array(sum(confusion_label, []))

    # Calculate metrics
    acc = accuracy_score(confusion_label, confusion_pred)
    f1 = f1_score(confusion_label, confusion_pred, average='weighted')

    # Log metrics to TensorBoard
    writer.add_scalar('Test/Loss', test_loss, epoch_num)
    writer.add_scalar('Test/Accuracy', acc, epoch_num)
    writer.add_scalar('Test/F1-score', f1, epoch_num)

    logger.debug(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Test Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Test Avg loss: {test_loss:>8f} \n")
    logger.debug(confusion_matrix(confusion_label, confusion_pred))
    print(confusion_matrix(confusion_label, confusion_pred))

    return 100 * correct, test_loss

def evaluate(device, epoch_num, logger, writer, dataloader, model, loss_fn, scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    valid_loss, correct = 0, 0
    confusion_pred, confusion_label =[], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embeddings, logits = model(device, X)
            # sum up the test loss per batch
            valid_loss += loss_fn(output=logits, embeddings=embeddings, labels=y.long().squeeze()).item()
            # sum up the number of correct predictions per batch
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()
            # predicted labels [Type: A list of lists, list of batches of predictions]
            confusion_pred.append(list(logits.argmax(1).cpu().numpy()))
            # print("shape of logits: ", logits.shape)

            # actual labels [Type: A list of lists, list of batches of labels]
            confusion_label.append(list(y.cpu().numpy()))
    scheduler.step()
    # loss of each batch
    valid_loss /= num_batches
    # accuracy rate
    correct /= size
    # flatten the list of lists into an array
    confusion_pred = np.array(sum(confusion_pred, []))
    # print("predictions: ",confusion_pred)
    # flatten the list of lists into an array
    confusion_label = np.array(sum(confusion_label, []))

    # Calculate metrics
    acc = accuracy_score(confusion_label, confusion_pred)
    f1 = f1_score(confusion_label, confusion_pred, average='weighted')

    # Log metrics to TensorBoard
    writer.add_scalar('Validate/Loss', valid_loss, epoch_num)
    writer.add_scalar('Validate/Accuracy', acc, epoch_num)
    writer.add_scalar('Validate/F1-score', f1, epoch_num)

    logger.debug(f"Valid Error: \n Accuracy: {(100*acc):>0.1f}%, Valid Avg loss: {valid_loss:>8f} \n")
    print(f"Valid Error: \n Accuracy: {(100*acc):>0.1f}%, Valid Avg loss: {valid_loss:>8f} \n")
    logger.debug(confusion_matrix(confusion_label, confusion_pred))
    print(confusion_matrix(confusion_label, confusion_pred))

    return 100 * correct, valid_loss


def main():
    # Set up logging and devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log1 = get_logger('log1', './model/weight/log.txt', logging.DEBUG)
    log2 = get_logger('log2', './model/weight/pltdata.txt', logging.DEBUG)

    # Set up arguments
    args = get_train_args()
    args.save_dir = util.get_save_dir(args.save_dir, args.name)
    # TensorBoard writer
    writer = SummaryWriter(args.save_dir)

    # Check the current arguments to choose model
    print("The training weights of this model is saved at ", args.save_dir, '/train/', args.name,
          "\n The current `load path` is: ", args.load_path,
          "\n The current boolean `freeze` is: ", args.freeze,
          "\n The current boolean `training` is: ", args.training,
          "\n The current boolean `attention` is: ", args.attention,
          "\n The current int `epochs` is: ", args.epochs,
          "\n The current str `model_name` is: ", args.model_name,
          "\n The current str `loss_name` is: ", args.loss_name, 
          "\n The current str `augment` is: ", args.augment,"\n")

    # Data pre-processor
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Data pre-processor
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate the image within a range of -30 to 30 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    # Dataset loader
    root_dir = '/content/gdrive/MyDrive/Dissertation/dataset/Dog Emotion/images'
    train_df, valid_df, test_df = split_dataset(root_dir=root_dir)
    # Augment train and validate dataset if needed
    if str2bool(args.augment) == True:
      train_df = augment(df=train_df,transform=train_transform, dataset="train")
      valid_df = augment(df=valid_df,transform=train_transform, dataset="validate")
    
    training_data = DogExpressionDataset(args=args, root_dir=root_dir, df=train_df, transform=test_transform, dataset="Train")
    valid_data = DogExpressionDataset(args=args, root_dir=root_dir, df=valid_df, transform=test_transform, dataset="Validate")
    test_data = DogExpressionDataset(args=args, root_dir=root_dir, df=test_df, transform=test_transform, dataset="Test")

    train_loader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    print(args.model_name)

    if args.model_name == "VIT_patch16":
        """"
        args:
        
        str2bool(args.freeze):     [boolean] freeze pre-trained or not
        """
        model2train = VIT_patch16(device, str2bool(args.freeze))

    elif args.model_name == "VIT_patch32":
        """"
        args:
            str2bool(args.freeze):     [boolean] freeze pre-trained or not
        """
        model2train = VIT_patch32(device, str2bool(args.freeze))
    elif args.model_name == "ResNet_50":
        """"
            args:
                str2bool(args.freeze):     [boolean] freeze pre-trained or not
        """
        model2train = ResNet_50(device, str2bool(args.freeze))
    elif args.model_name == "ResNet_34":
        """"
                args:
                    str2bool(args.freeze):     [boolean] freeze pre-trained or not
                """
        model2train = ResNet_34(device, str2bool(args.freeze))

    elif args.model_name == "ResVit":
        """"
                args:

                str2bool(args.freeze):     [boolean] freeze pre-trained or not
                """
        model2train = ResVit(device, args=args)

    else:
        print("Sorry, there isn't a model related to this model, please make sure you choose the correct model name")



    # Load weights if needed
    if args.load_path:
        model2train.load_model(args.load_path)  # args.load_path [str][default:None] path to load the weights

    # Define loss and optimizer
    if args.loss_name == "Cross_Entropy":
        loss = HybridLoss(device, args, temperature=1, alpha=1) 
    elif args.loss_name == "Hybrid_loss":
        loss = HybridLoss(device, args, temperature=0.3, alpha=0.5)
        print("temperature", 0.3)
    else:
        raise AssertionError("No such loss, it has to be either Cross_Entropy or Hybrid_loss")
        
    optimizer = torch.optim.AdamW(model2train.parameters(), lr=0.00001) # weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.2) + 0.2)
    best_correct = 0

    # Whether to train the whole network
    if str2bool(args.training):
        for t in range(args.epochs):
            print(f"Epoch {t + 1}\n-------------------------------\n")
            log1.info(f"Epoch {t + 1}\n-------------------------------\n")
            train(device, t, log1, writer, train_loader, model2train, loss, optimizer)
            test_correct, test_loss = test(device, t, log1, writer, test_loader, model2train, loss, scheduler)
            valid_correct, valid_loss = evaluate(device, t, log1, writer, valid_loader, model2train, loss, scheduler)
            log2.info(str(t) + '___' + str(valid_loss) + '___' + str(valid_correct) + str(test_loss) + str(test_correct) + '\n')
            if best_correct < test_correct:
                best_correct = test_correct
                model2train.save_model(args.save_dir)  # updated args.save_dir at the beginning of the main()
                # torch.save(model2train.state_dict(), "./model/weight/VIT_patch16_epoch.pth")
                log1.info("Saved PyTorch Model State epoch is " + str(t) + "   valid_correct = " + str(valid_correct) + "   test_correct = " + str(test_correct) + "\n")
                print("Saved PyTorch Model State epoch is " + str(t) + "   valid_correct = " + str(valid_correct) + "   test_correct = " + str(test_correct) + "\n")
        print("The best accuracy is :", best_correct)
    else:
        """
        Please remember to load a path of weights if you do not want to train

        The weights are in the `weights` folder
        """
        # Check if there is a path of weights given for loading into the model
        assert args.load_path, "\n Please load the weights if you don't wanna train the whole model. " \
                                "\n Please check more details in args.py"
        correct, test_loss = test(device, 0, log1, writer, test_loader, model2train, loss, scheduler)
        model2train.save_model(args.save_dir)  # updated args.save_dir at the beginning of the main()
        # torch.save(model2train.state_dict(), "./model/weight/VIT_patch16_epoch.pth")
        log1.info("Load trained model without training with epoch " + "t=0" + "   correct = " + str(correct) + "\n")
        print("Load trained model without training with epoch " + "t=0" + "   correct = " + str(correct) + "\n")

    log1.info("Done!\n")
    print("Done!\n")

if __name__ == '__main__':

    main()
