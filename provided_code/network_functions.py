""" This class inherits a network architecture and performs various functions on a define architecture like training
 and predicting"""

import os

import numpy as np
import pandas as pd
import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from provided_code.general_functions import get_paths, make_directory_and_return_path, sparse_vector_function
from provided_code.network_architectures import DefineDoseFromCT


class PredictionModel(DefineDoseFromCT):

    def __init__(self, data_loader, results_patent_path, model_name, stage='training'):
        """
        Initialize the Prediction model class
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        """
        # set attributes for data shape from data loader
        self.data_loader = data_loader
        self.patient_shape = data_loader.patient_shape
        self.full_roi_list = data_loader.full_roi_list
        self.model_name = model_name

        # Define training parameters
        self.epoch_start = 0  # Minimum epoch (overwritten during initialization if a newer model exists)
        self.epoch_last = 200  # When training will stop

        # Define image sizes
        self.dose_shape = (*self.patient_shape, 1)
        self.ct_shape = (*self.patient_shape, 1)
        self.roi_masks_shape = (*self.patient_shape, len(self.full_roi_list))

        # Define filter and stride lengths
        self.filter_size = (4, 4, 4)
        self.stride_size = (2, 2, 2)

        # Define the initial number of filters in the model (first layer)
        self.initial_number_of_filters = 1  # 64

        # Define model optimizer
        self.gen_optimizer = Adam(lr=0.0002, decay=0.001, beta_1=0.5, beta_2=0.999)

        # Define place holders for model
        self.generator = None

        # Make directories for data and models
        model_results_path = '{}/{}'.format(results_patent_path, model_name)
        self.model_dir = make_directory_and_return_path('{}/models'.format(model_results_path))
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, stage)

        # Make template for model path
        self.model_path_template = '{}/epoch_'.format(self.model_dir)

    def train_model(self, epochs=200, save_frequency=5, keep_model_history=2):
        """
        Train the model over several epochs
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        :return: None
        """
        # Define new models, or load most recent model if model already exists
        self.epoch_last = epochs
        self.initialize_networks()

        # Check if training has already finished
        if self.epoch_start == epochs:
            return

        else:
            # Start training GAN
            num_batches = self.data_loader.number_of_batches()
            for e in range(self.epoch_start, epochs):
                # Begin a new epoch
                print('epoch number {}'.format(e))
                self.data_loader.on_epoch_end()  # Shuffle the data after each epoch
                for i in tqdm.tqdm(range(num_batches)):
                    # Load a subset of the data and train the network with the data
                    self.train_network_on_batch(i, e)

                # Create epoch label and save models at the specified save frequency
                current_epoch = e + 1
                if 0 == np.mod(current_epoch, save_frequency):
                    self.save_model_and_delete_older_models(current_epoch, save_frequency, keep_model_history)

    def save_model_and_delete_older_models(self, current_epoch, save_frequency, keep_model_history):
        """
        Save the current model and delete old models, based on how many models the user has asked to keep. We overwrite
        files (rather than deleting them) to ensure the user's trash doesn't fill up.
        :param current_epoch: the current epoch number that is being saved
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        """

        # Save the model to a temporary path
        temporary_model_path = '{}_temp.h5'.format(self.model_path_template)
        self.generator.save(temporary_model_path)
        # Define the epoch that should be over written
        epoch_to_overwrite = current_epoch - keep_model_history * save_frequency
        # Make appropriate path to save model at
        if epoch_to_overwrite > 0:
            model_to_delete_path = '{}{}.h5'.format(self.model_path_template, epoch_to_overwrite)
        else:
            model_to_delete_path = '{}{}.h5'.format(self.model_path_template, current_epoch)
        # Save model
        os.rename(temporary_model_path, model_to_delete_path)
        # The code below is a hack to ensure the Google Drive trash doesn't fill up
        if epoch_to_overwrite > 0:
            final_save_model_path = '{}{}.h5'.format(self.model_path_template, current_epoch)
            os.rename(model_to_delete_path, final_save_model_path)

    def initialize_networks(self):
        """
        Load the newest model, or if no model exists with the appropriate name a new model will be created.
        :return:
        """
        # Initialize variables for models
        all_models = get_paths(self.model_dir, ext='h5')

        # Get last epoch of existing models if they exist
        for model_name in all_models:
            model_epoch_number = model_name.split(self.model_path_template)[-1].split('.h5')[0]
            if model_epoch_number.isdigit():
                self.epoch_start = max(self.epoch_start, int(model_epoch_number))

        # Build new models or load most recent old model if one exists
        if self.epoch_start >= self.epoch_last:
            print('Model fully trained, loading model from epoch {}'.format(self.epoch_last))
            return 0, 0, 0, self.epoch_last

        elif self.epoch_start >= 1:
            # If models exist then load them
            self.generator = load_model('{}{}.h5'.format(self.model_path_template, self.epoch_start))
        else:
            # If models don't exist then define them
            self.define_generator()

    def train_network_on_batch(self, batch_index, epoch_number):
        """Loads a sample of data and uses it to train the model
        :param batch_index: The batch index
        :param epoch_number: The epoch
        """
        # Load images
        image_batch = self.data_loader.get_batch(batch_index)

        # Train the generator model with the batch
        model_loss = self.generator.train_on_batch([image_batch['ct'], image_batch['structure_masks']],
                                                   image_batch['dose'])

        print('Model loss at epoch {} batch {} is {:.3f}'.format(epoch_number, batch_index, model_loss))

    def predict_dose(self, epoch=1):
        """Predicts the dose for the given epoch number, this will only work if the batch size of the data loader
        is set to 1.
        :param epoch: The epoch that should be loaded to make predictions
        """
        # Define new models, or load most recent model if model already exists
        self.generator = load_model('{}{}.h5'.format(self.model_path_template, epoch))
        os.makedirs(self.prediction_dir, exist_ok=True)
        # Use generator to predict dose
        number_of_batches = self.data_loader.number_of_batches()
        print('Predicting dose')
        for idx in tqdm.tqdm(range(number_of_batches)):
            image_batch = self.data_loader.get_batch(idx)

            # Get patient ID and make a prediction
            pat_id = image_batch['patient_list'][0]
            dose_pred_gy = self.generator.predict([image_batch['ct'], image_batch['structure_masks']])
            dose_pred_gy = dose_pred_gy * image_batch['possible_dose_mask']
            # Prepare the dose to save
            dose_pred_gy = np.squeeze(dose_pred_gy)
            dose_to_save = sparse_vector_function(dose_pred_gy)
            dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                   columns=['data'])
            dose_df.to_csv('{}/{}.csv'.format(self.prediction_dir, pat_id))
