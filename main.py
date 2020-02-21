import shutil

from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import EvaluateDose
from provided_code.general_functions import get_paths, make_directory_and_return_path
from provided_code.network_functions import PredictionModel

if __name__ == '__main__':
    # drive.mount('/content/drive')

    # Define parent directory
    training_data_dir = '/Users/aaronbabier/FINAL-3/train-pats'
    validation_data_dir = '/Users/aaronbabier/FINAL-3/valid-pats'
    results_dir = '/Users/aaronbabier/open-kbp-results'  # parent path where results are stored
    prediction_name = 'baseline'
    number_of_training_epochs = 1
    # Prepare the data
    plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
    num_train_pats = 2  # number of plans that will be used to train model
    training_paths = plan_paths[:num_train_pats]  # list of training plans
    hold_out_paths = plan_paths[num_train_pats:5]  # list of paths used for held out testing

    # Train a model
    data_loader_train = DataLoader(training_paths)
    dose_prediction_model_train = PredictionModel(data_loader_train, results_dir, model_name=prediction_name)
    dose_prediction_model_train.train_model(epochs=number_of_training_epochs, save_frequency=1, keep_model_history=1)

    # Predict dose for the held out set
    data_loader_hold_out = DataLoader(hold_out_paths, mode_name='dose_prediction')
    dose_prediction_model_hold_out = PredictionModel(data_loader_hold_out, results_dir, model_name=prediction_name)
    # dose_prediction_model_hold_out.predict_dose(epoch=number_of_training_epochs)

    # Evaluate dose metrics
    # data_loader_hold_out_eval = DataLoader(hold_out_paths, mode_name='evaluation')  # Set data loader
    # prediction_paths = get_paths(dose_prediction_model_hold_out.prediction_dir, ext='csv')
    # hold_out_prediction_loader = DataLoader(prediction_paths, mode_name='predicted_dose')  # Set prediction loader
    # dose_evaluator = EvaluateDose(data_loader_hold_out_eval, hold_out_prediction_loader)
    # dvh_score, dose_score = dose_evaluator.make_metrics()
    # print('In this out-of-sample test:\n'
    #       '\tthe DVH score is {:.3f}\n '
    #       '\tthe dose score is {:.3f}'.format(dvh_score, dose_score))

    # Apply model to validation set
    validation_data_paths = get_paths(validation_data_dir, ext='')  # gets the path of each plan's directory
    validation_data_loader = DataLoader(validation_data_paths, mode_name='dose_prediction')
    dose_prediction_model_validation = PredictionModel(validation_data_loader, results_dir,
                                                       model_name=prediction_name, stage='validation')
    dose_prediction_model_validation.predict_dose(epoch=number_of_training_epochs)

    # Evaluate plans based on dose metrics (no baseline available to compare to)
    validation_eval_data_loader = DataLoader(validation_data_paths, mode_name='evaluation')  # Set data loader
    dose_evaluator = EvaluateDose(validation_eval_data_loader)
    dose_evaluator.make_metrics()
    validation_prediction_metrics = dose_evaluator.reference_dose_metric_df.head()

    # Zip dose to submit
    submission_dir = make_directory_and_return_path('{}/submissions'.format(results_dir))
    shutil.make_archive('{}/{}'.format(submission_dir, prediction_name), 'zip', dose_prediction_model_validation.prediction_dir)
