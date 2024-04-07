import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, AdamW, GPT2ForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import EMG_utils
import pickle
import datetime
from EMG_ENV import ENV


def validation_loop(num_layers_to_train,lr_list,train_loader,Valid_loader,train_dataset):
    """
        Executes a validation loop for training GPT-2 models across specified hyperparameters, aiming to identify
        optimal configurations based on training and validation performance. It iterates over combinations of the
        number of transformer layers to train and learning rates, training a new model for each combination.

        For each hyperparameter set, the function trains the model, evaluates it against both training and validation
        datasets, and logs performance metrics such as accuracy, loss, and confusion matrix. Each trained model is saved
        to disk, and performance metrics are visualized and stored in a dictionary.

        Parameters:
        - num_layers_to_train (list of int): A list specifying the number of top transformer layers of GPT-2 to fine-tune.
        - lr_list (list of float): A list of learning rates to use for training the models.
        - train_loader (DataLoader): DataLoader for the training dataset.
        - Valid_loader (DataLoader): DataLoader for the validation dataset.
        - train_dataset: Dataset used for training, required for some utilities like visualization.

        Returns:
        - all_results_dict (dict): A dictionary where each key represents a combination of learning rate and number of layers
          trained, and its value is another dictionary containing 'val_accuracy_list', 'train_accuracy_list',
          'confusion_matrix' for that configuration.
    """

    all_results_dict = {}
    for num_layers in num_layers_to_train:
        for lr in lr_list:
            results_dict = {}
            config = GPT2Config.from_pretrained(ENV.MODEL_NAME, ENV.NUM_LABELS)
            config.problem_type = ENV.PROBLEM_TYPE
            model = GPT2ForSequenceClassification.from_pretrained(ENV.MODEL_NAME, config=config)
            model.config.pad_token_id = tokenizer.pad_token_id
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            num_epochs = ENV.NUM_EPOCHS_FOR_VALIDATION
            for param in model.transformer.h[num_layers:].parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=lr)
            print(f"starts Val loop with lr={lr}, num layers = {num_layers}")
            val_losses, val_accuracy_list, train_losses, train_accuracy_list, confusion_matrix = EMG_utils.train_model(
                model, train_loader, Valid_loader, optimizer, num_epochs, device, train_dataset)
            dic_key = 'lr=' + str(lr) + ' num layers=' + str(num_layers)
            model_name = ENV.MODEL_SAVING_DIR + '/Val_models/lr=' + str(lr) + ' num layers=' + str(
                num_layers) + ' model'
            results_dict['val_accuracy_list'] = test_accuracy_list
            results_dict['train_accuracy_list'] = train_accuracy_list
            results_dict['confusion_matrix'] = confusion_matrix
            all_results_dict[dic_key] = results_dict
            model.save_pretrained(model_name)
            EMG_utils.show_confusion_matrix(val_accuracy_list[-1], confusion_matrix, label_range=ENV.NUM_LABELS,
                                            save=True, name=dic_key)
            EMG_utils.show_accuracy_graphs(val_losses, val_accuracy_list, train_losses, train_accuracy_list, save=True,
                                           name=dic_key)

    with open(ENV.VAL_DICT_SAVE_NAME, 'wb') as f:
        pickle.dump(all_results_dict, f)

    return all_results_dict


def train_model(model, train_loader, valid_loader, optimizer, num_epochs, device, train_dataset):
    """
        Trains and evaluates a machine learning model using specified training and validation datasets, optimizer,
        and the number of epochs. It calculates and logs training and validation loss and accuracy over each epoch.
        Before starting the training loop, it evaluates the model's initial performance on the validation dataset.

        Parameters:
        - model: The machine learning model to be trained and evaluated. This should be a PyTorch model instance.
        - train_loader (DataLoader): DataLoader for the training dataset, providing batches of data.
        - valid_loader (DataLoader): DataLoader for the validation dataset, used for evaluating model performance.
        - optimizer: The optimization algorithm used to update model weights
        - num_epochs (int): The number of training epochs to complete.
        - device: The device (CPU or GPU) to use for training and evaluation.
        - train_dataset: The training dataset used for calculating training loss. This is necessary for normalizing
          the loss across all training samples.

        Returns:
        A tuple containing lists of validation losses, validation accuracy, training losses, training accuracy, and the
        last confusion matrix calculated from the validation dataset. Specifically:
        - test_losses: List of float, validation losses after each epoch.
        - test_accuracy_list: List of float, validation accuracy values after each epoch.
        - train_losses: List of float, training losses after each epoch.
        - train_accuracy_list: List of float, training accuracy values after each epoch.
        - confusion_matrix: The last confusion matrix calculated from the validation dataset.
    """
    train_losses = []
    test_losses = []
    train_accuracy_list = []
    test_accuracy_list = []
    model_accuracy, confusion_matrix, test_loss = calculate_accuracy(model, valid_loader, device)
    print(f'accuracy before training is {model_accuracy}')
    test_losses.append(test_loss)
    test_accuracy_list.append(model_accuracy)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total, correct = 0, 0
        for i_batch, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)#FIXME: yuval - check positional embedings
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels.long())
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels.long()).sum().item()
            try:
                total += labels.size(0)
            except:
                total += 1
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        train_accuracy_list.append(100 * correct / total)

        # Evaluate model performance
        model_accuracy, confusion_matrix, test_loss = calculate_accuracy(model, valid_loader, device)
        print(f'accuracy for epoch {epoch} is {model_accuracy}')
        test_losses.append(test_loss)
        test_accuracy_list.append(model_accuracy)
    return test_losses,test_accuracy_list,train_losses,train_accuracy_list,confusion_matrix

def calculate_accuracy(model, dataloader, device):
    """
    Calculates the accuracy and confusion matrix of a given model on provided data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset for evaluation.
        device (torch.device): The device on which to perform calculations (e.g., 'cuda' or 'cpu').

    Returns:
        float: The accuracy of the model on the given dataset as a percentage.
        np.array: A confusion matrix representing model predictions.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    total_correct = 0
    total_data = 0
    test_loss = 0.0
    confusion_matrix = np.zeros([10, 10], int)  # Adjust size according to the number of classes.
    with torch.no_grad():  # No gradients required for evaluation.
        for batch in dataloader:
            # Move input and labels to the specified device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

            # Update totals
            total_data += labels.size(0)
            total_correct += (predictions == labels).sum().item()

            loss = outputs.loss
            test_loss += loss.item() * input_ids.size(0)

            # Update confusion matrix
            for i, l in enumerate(labels):
                confusion_matrix[int(l.item()), predictions[i].item()] += 1

    model_accuracy = total_correct / total_data * 100
    test_loss /= total_data
    return model_accuracy, confusion_matrix, test_loss


def show_confusion_matrix(test_accuracy, confusion_matrix, label_range, save = False, name = None):
    """
    Displays the confusion matrix and test accuracy.

    Args:
        test_accuracy (float): The accuracy of the model on the test set.
        confusion_matrix (np.array): A confusion matrix of the model's predictions.
        label_range (int): The number of unique labels in the dataset.
    """
    print(f"Test accuracy: {test_accuracy:.3f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(confusion_matrix, aspect='auto', vmin=0, cmap=plt.get_cmap('Blues'))
    fig.colorbar(cax)
    plt.ylabel('Actual Category')
    plt.yticks(range(label_range))
    plt.xlabel('Predicted Category')
    plt.xticks(range(label_range))
    plt.show()
    if save:
        saving_name = '/home/yuval/om2seq/EMG/graphs/matrix '+name+'.png'
        plt.savefig(saving_name)


def show_accuracy_graphs(test_losses, test_accuracy_list, train_losses, train_accuracy_list, save = False, name = None):
    """
    Plots the training and testing loss and accuracy over epochs.

    Args:
        test_losses (list): List of test losses over epochs.
        test_accuracy_list (list): List of test accuracies over epochs.
        train_losses (list): List of training losses over epochs.
        train_accuracy_list (list): List of training accuracies over epochs.
    """


    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(train_losses, 'r', label='Train loss')  # Corrected label
    plt.plot(test_losses, 'b', label='Test loss')  # Corrected label

    plt.legend()
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        saving_name = '/home/yuval/om2seq/EMG/graphs/loss '+name+'.png'
        plt.savefig(saving_name)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plt.plot(train_accuracy_list, 'r', label='Train accuracy')  # Corrected label
    plt.plot(test_accuracy_list, 'b', label='Test accuracy')  # Corrected label

    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if save:
        saving_name = '/home/yuval/om2seq/EMG/graphs/acc '+name+'.png'
        plt.savefig(saving_name)
    plt.show()

