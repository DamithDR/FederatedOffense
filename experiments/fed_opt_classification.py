import argparse
import logging
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch.cuda
from pandas import DataFrame
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer

from config.training_args import train_args
from fuse.main_model_fuse import ModelLoadingInfo, load_model, fuse_models
from utils.label_converter import encode, decode
from utils.print_stat import print_information


# import torch.multiprocessing

# otherwise the shared memory is not enough, so it throws an error
# torch.multiprocessing.set_sharing_strategy('file_system')

# eddited the file to do fed_opt experiments
def run():
    logging.basicConfig(filename='3m777r-results.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='''fuse multiple models ''')
    parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
    parser.add_argument('--n_fold', required=False, help='n_fold predictions', default=5)
    parser.add_argument('--train_base_models', required=False, help='train base models', default=True, type=bool)
    parser.add_argument('--base_model_paths', required=False, help='base model paths', default='')
    parser.add_argument('--datasets', required=True, help='comma separated datasets')
    parser.add_argument('--base_model_type', required=False, help='Base model type', default='bert')
    parser.add_argument('--base_model', required=False, help='n_fold predictions',
                        default='bert-large-cased')

    arguments = parser.parse_args()
    datasets = arguments.datasets.split(',')
    n_fold = int(arguments.n_fold)
    base_model = arguments.base_model
    base_model_type = arguments.base_model_type

    print('data loading started')
    if torch.cuda.is_available():
        torch.device('cuda')
        torch.cuda.set_device(int(arguments.device_number))
    # ========================================================================

    print('training started')
    model_paths = []
    train_sets = []
    test_sets = []
    eval_sets = []
    df_finetune = DataFrame()

    for dataset in datasets:
        dataset = str(dataset).lower()
        train_file = 'data/' + dataset + '/' + dataset + '_train.csv'
        test_file = 'data/' + dataset + '/' + dataset + '_test.csv'
        train = pd.read_csv(train_file, sep='\t')
        train = train.rename(columns={'Text': 'text', 'Class': 'labels'})
        # train['labels'] = encode(train['labels'])
        label_lst = ['OFF', 'NOT']
        category2id = {cat: idx for idx, cat in enumerate(label_lst)}
        id2category = {idx: cat for idx, cat in enumerate(label_lst)}
        train['labels'] = train['labels'].apply(lambda x: category2id[x])
        train, dev = train_test_split(train, test_size=0.2, random_state=777)

        # if finetune_dataset.__eq__(dataset):
        # anyway we are going to have 2 clients
        train, df_finetune = train_test_split(train, test_size=0.2, random_state=777)
        df_finetune.to_csv(f'ft_{dataset}.csv', sep='\t', index=False)
        train_sets.append(train)
        eval_sets.append(dev)
        test = pd.read_csv(test_file, sep='\t')
        test = test.rename(columns={'Text': 'text', 'Class': 'labels'})
        test_sets.append(test)
        model_path = 'model_' + dataset
        model_paths.append(model_path)

    if arguments.train_base_models:
        print('training base models')
        for model_path, df_train, df_eval in zip(model_paths, train_sets, eval_sets):
            train_args['best_model_dir'] = model_path
            model = ClassificationModel(
                base_model_type, base_model, use_cuda=torch.cuda.is_available(),
                args=train_args
            )
            model.train_model(df_train, eval_df=df_eval)

        print('models training finished')
    else:
        print('model training skipped')
        model_paths = str(arguments.base_model_paths).split(',')

    # ========================================================================

    # model_paths = ['model_davidson/', 'model_olid/']
    # fusing multiple models
    # print('model fusing started')
    # model_info = ModelLoadingInfo(name=base_model, tokenizer_name=base_model,
    #                               classification=True)
    # models_to_fuse = [ModelLoadingInfo(name=model, tokenizer_name=model, classification=True) for model in model_paths]
    # base_model = load_model(model_info)
    # fused_model = fuse_models(base_model, models_to_fuse)
    # # saving fused model for predictions
    # fused_model.save_pretrained(train_args['fused_model_path'])
    # tokenizer = AutoTokenizer.from_pretrained(model_paths[0])  # get the 1st model path to get the tokenizer
    # tokenizer.save_pretrained(train_args['fused_model_path'])
    # print('fused model saved')

    # # load the saved model
    # train_args['best_model_dir'] = train_args['fused_finetuned_model_path']
    # general_model = ClassificationModel(
    #     base_model_type, train_args['fused_model_path'], use_cuda=torch.cuda.is_available(), args=train_args
    # )
    #
    # # finetune model
    # df_finetune, df_eval = train_test_split(df_finetune, test_size=0.1)
    # general_model.train_model(df_finetune, eval_df=df_eval)
    #
    # fine_tuned_model = general_model  # to use directly
    #
    # # fine_tuned_model = ClassificationModel(
    # #     "bert", train_args['fused_finetuned_model_path'], use_cuda=torch.cuda.is_available(), args=train_args
    # # )
    #
    # print('Starting Predictions')
    # macros = []
    # micros = []
    #
    # with open('out.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         i = 0
    #         for df_test, dataset in zip(test_sets, datasets):
    #             test_preds = np.zeros((len(df_test), n_fold))
    #             for fold in range(0, n_fold):
    #                 # predictions
    #                 print('Starting Prediction fold no : ' + str(fold))
    #
    #                 predictions, raw_outputs = fine_tuned_model.predict(df_test['text'].tolist())
    #
    #                 test_preds[:, fold] = predictions
    #                 print("Completed Fold {}".format(fold))
    #
    #             final_predictions = []
    #             for row in test_preds:
    #                 row = row.tolist()
    #                 final_predictions.append(int(max(set(row), key=row.count)))
    #
    #             df_test['prediction'] = final_predictions
    #             df_test['prediction'] = decode(df_test['prediction'])
    #             print(f'Results for dataset : {dataset} for fold {n_fold} : ')
    #             macro_f1, micro_f1 = print_information(df_test, 'prediction', 'labels')
    #             df_test.to_csv('predictions_vs_test' + str(dataset) + str(i) + '.csv', index=False)
    #             i += 1
    #             macros.append(macro_f1)
    #             micros.append(micro_f1)
    #
    #             print(f'Final Results for dataset : {dataset}')
    #             print('=====================================================================')
    #
    #             macro_str = "Macro F1 Mean - {} | STD - {}\n".format(np.mean(macros), np.std(macros))
    #             micro_str = "Micro F1 Mean - {} | STD - {}".format(np.mean(micros), np.std(micros))
    #             print(macro_str)
    #             print(micro_str)
    #
    #             print('======================================================================')
    #         print(f"fused model finetuned on dataset : {finetune_dataset}")
    print("Done")


if __name__ == '__main__':
    run()
