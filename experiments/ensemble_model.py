import argparse
import logging
import os.path
from contextlib import redirect_stdout

import pandas as pd
import torch.cuda
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

from config.training_args import train_args
from utils.label_converter import encode, decode


# import torch.multiprocessing

# otherwise the shared memory is not enough, so it throws an error
# torch.multiprocessing.set_sharing_strategy('file_system')

def run():
    logging.basicConfig(filename='3m777r-results.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='''fuse multiple models ''')
    parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
    parser.add_argument('--n_fold', required=False, help='n_fold predictions', default=1)
    parser.add_argument('--base_model_type', required=True, help='Base model type')
    parser.add_argument('--base_model', required=False, help='n_fold predictions',
                        default='bert-large-cased')

    arguments = parser.parse_args()
    n_fold = int(arguments.n_fold)
    base_model = arguments.base_model
    base_model_type = arguments.base_model_type

    model_paths = {}
    test_sets = {}
    datasets = ['hasoc', 'hatexplain', 'olid','davidson']
    print('training started')
    for dataset in datasets:
        train_file = 'data/' + dataset + '/' + dataset + '_train.csv'
        test_file = 'data/' + dataset + '/' + dataset + '_test.csv'
        train = pd.read_csv(train_file, sep='\t')
        train = train.rename(columns={'Text': 'text', 'Class': 'labels'})
        train['labels'] = encode(train['labels'])
        train, eval_set = train_test_split(train, test_size=0.2, random_state=777)
        test = pd.read_csv(test_file, sep='\t')
        test = test.rename(columns={'Text': 'text', 'Class': 'labels'})
        test_sets[dataset] = test
        model_path = 'ensemble_models/' + dataset + '/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_paths[dataset] = model_path

        # for model_path, df_train, df_eval in zip(model_paths, train_sets, eval_sets):
        train_args['best_model_dir'] = model_path

        model = ClassificationModel(
            base_model_type, base_model, use_cuda=torch.cuda.is_available(),
            args=train_args
        )
        model.train_model(train, eval_df=eval_set)

        print(f'model creation done for : {dataset}')
    print("training finished")
    print('Starting Predictions')

    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            i = 0
            for saved_model in model_paths:
                prediction_model = ClassificationModel(
                    base_model_type, saved_model, use_cuda=torch.cuda.is_available(),
                    args=train_args
                )
                for dataset in datasets:
                    df_test = test_sets[dataset]
                    for fold in range(0, n_fold):
                        # predictions
                        print('Starting Prediction fold no : ' + str(fold))
                        predictions, raw_outputs = prediction_model.predict(df_test['text'].tolist())

                        print("Completed Fold {}".format(fold))
                        df_test['predictions'] = decode(predictions)
                        filename = "ensemble_results/" + "M-" + saved_model.split('/')[1] + "D-" + dataset + ".csv"
                        df_test.to_csv(filename, sep='\t',index=False)

                    print('======================================================================')
                print(f"Predictions Generated on model {saved_model.split('/')[1]} dataset : {dataset}")
    print("Done")


if __name__ == '__main__':
    run()
