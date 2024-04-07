import pandas as pd
import torch
import os
import random
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, AdamW, GPT2ForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import EMG_utils  # Ensure this module has all the required functions
import pickle
import datetime
from EMG_ENV import ENV
from EMG_DATA_LOADER import Data_loader

train_loader = Data_loader.train_loader
test_loader = Data_loader.test_loader
Valid_loader = Data_loader.Valid_loader

config = GPT2Config.from_pretrained(ENV.MODEL_NAME, ENV.NUM_LABELS)
config.problem_type = ENV.PROBLEM_TYPE
model = GPT2ForSequenceClassification.from_pretrained(ENV.MODEL_NAME, config=config)
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = ENV.NUM_EPOCHS

# Validation loop
# declare hyperparameter search
num_layers_to_train = [-12,-9,-6,-3]
lr_list = [1e-5, 5e-5, 1e-4]

all_results_val_dict=validation_loop(num_layers_to_train,lr_list,train_loader,Valid_loader,train_dataset)

# create  model
config = GPT2Config.from_pretrained(ENV.MODEL_NAME, ENV.NUM_LABELS)
config.problem_type = ENV.PROBLEM_TYPE
model = GPT2ForSequenceClassification.from_pretrained(ENV.MODEL_NAME, config=config)
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = ENV.NUM_EPOCHS

# Fine-tuning last layers
for param in model.transformer.h[ENV.NUM_LAYERS:].parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=ENV.lr)  # Adjust learning rate as needed

# Training loop
print("starts the training")
test_losses, test_accuracy_list, train_losses, train_accuracy_list, confusion_matrix = EMG_utils.train_model(model,
                                                                                                             train_loader,
                                                                                                             test_loader,
                                                                                                             optimizer,
                                                                                                             num_epochs,
                                                                                                             device,
                                                                                                             train_dataset)

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Visualize model performance
EMG_utils.show_confusion_matrix(test_accuracy_list[-1], confusion_matrix, label_range=ENV.NUM_LABELS, save=True,
                                name=ENV.CONFUSION_MATRIX_NAME + current_time)
EMG_utils.show_accuracy_graphs(test_losses, test_accuracy_list, train_losses, train_accuracy_list, save=True,
                               name=ENV.ACC_GRAPH_NAME + current_time)

# Save the fine-tuned model
model.save_pretrained(ENV.MODEL_SAVING_NAME + current_time)

a = 5

results_dict = {}
results_dict['test_accuracy_list'] = test_accuracy_list
results_dict['train_accuracy_list'] = train_accuracy_list
results_dict['confusion_matrix'] = confusion_matrix
results_dict['test_losses'] = test_losses
results_dict['train_losses'] = train_losses

with open(ENV.RESULTS_DICT_SAVE_NAME + current_time + ".pkl", 'wb') as f:
    pickle.dump(results_dict, f)


