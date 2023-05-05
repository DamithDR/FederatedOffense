import argparse
import itertools
import os
import shutil
import sys

import pandas as pd
import torch

import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config.training_args import train_args
from fuse.main_model_fuse import ModelLoadingInfo, load_model, fuse_models
from utils.label_converter import decode
from utils.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''fuse multiple models ''')
parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
parser.add_argument('--n_fold', required=False, help='n_fold predictions', default=5)
parser.add_argument('--base_model_type', required=False, help='Base model type', default='bert')
parser.add_argument('--base_model', required=False, help='n_fold predictions',
                    default='bert-large-cased')
arguments = parser.parse_args()

base_models = ['model_davidson', 'model_olid', 'model_hasoc', 'model_hatexplain']
finetune_datasets = ['ft_davidson.csv', 'ft_olid.csv', 'ft_hasoc.csv', 'ft_hatexplain.csv']

temporary_path = 'temp_fused_models/'
temp_best_model_path = 'temp_best_model_path/'
results_path = 'predicted_results/'

for i in range(2, len(base_models)):
    combinations = itertools.combinations(base_models, i)
    for combination in combinations:
        model_paths = list(combination)
        print(f'combination :{list(combination)}')
        ft_datasets_list = ['ft_' + path.split('_')[1] + '.csv' for path in list(combination)]
        print(f'finetune : {ft_datasets_list}')
        test_datasets = ['data/' + name.split('_')[1] + '/' + name.split('_')[1] + '_test.csv' for name in
                         set(base_models) - set(combination)]
        print(test_datasets)

        print('model fusing started')
        model_info = ModelLoadingInfo(name=arguments.base_model, tokenizer_name=arguments.base_model,
                                      classification=True)
        models_to_fuse = [ModelLoadingInfo(name=model, tokenizer_name=model, classification=True) for model in
                          model_paths]
        base_model = load_model(model_info)
        fused_model = fuse_models(base_model, models_to_fuse)
        # saving fused model for predictions
        fused_model.save_pretrained(temporary_path)
        tokenizer = AutoTokenizer.from_pretrained(model_paths[0])  # get the 1st model path to get the tokenizer
        tokenizer.save_pretrained(temporary_path)
        print('fused model saved to temporary path')

        # load the saved model
        train_args['best_model_dir'] = temp_best_model_path
        for f_tune_file in ft_datasets_list:
            general_model = ClassificationModel(
                arguments.base_model_type, temporary_path, use_cuda=torch.cuda.is_available(),
                args=train_args
            )
            # finetune model
            df_finetune = pd.read_csv(f_tune_file, sep='\t')
            df_finetune, df_eval = train_test_split(df_finetune, test_size=0.1)
            general_model.train_model(df_finetune, eval_df=df_eval)

            fine_tuned_model = general_model  # to use directly

            print('Starting Predictions')

            save_file_name = results_path + '_'.join(combination) + '.txt'

            with open(save_file_name, 'w') as f:
                f.write(f'finetuned on {f_tune_file}\n')
                f.write('==============================================\n')
                macros = []
                micros = []
                for test_file in test_datasets:
                    f.write(f'test set :- {test_file}\n')
                    df_test = pd.read_csv(test_file, sep='\t')
                    test_preds = np.zeros((len(df_test), arguments.n_fold))
                    for fold in range(0, arguments.n_fold):
                        # predictions
                        print('Starting Prediction fold no : ' + str(fold))

                        predictions, raw_outputs = fine_tuned_model.predict(df_test['text'].tolist())

                        test_preds[:, fold] = predictions
                        print("Completed Fold {}".format(fold))

                        df_test['prediction'] = predictions
                        df_test['prediction'] = decode(df_test['prediction'])
                        macro_f1, micro_f1 = print_information(df_test, 'prediction', 'labels')
                        macros.append(macro_f1)
                        micros.append(micro_f1)

                    f.write('=====================================================================\n')

                    macro_str = "Macro F1 Mean - {} | STD - {}\n".format(np.mean(macros), np.std(macros))
                    micro_str = "Micro F1 Mean - {} | STD - {}\n".format(np.mean(micros), np.std(micros))
                    f.write(macro_str)
                    f.write(micro_str)

                    f.write('======================================================================\n')
            del general_model
            print("Done")

        shutil.rmtree(temporary_path + '*.*')
