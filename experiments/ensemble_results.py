import itertools

import pandas as pd

from utils.label_converter import encode
from utils.print_stat import print_information

datasets = ['hasoc', 'hatexplain', 'olid', 'davidson']
folder_paths = ['data/ensemble/results/fbert/', 'data/ensemble/results/bert-large/']

test_dict = {}

for dataset in datasets:
    test_file = 'data/' + dataset + '/' + dataset + '_test.csv'
    test = pd.read_csv(test_file, sep='\t')
    test = test.rename(columns={'Text': 'text', 'Class': 'labels'})
    test_dict[dataset] = test

for path in folder_paths:
    for L in range(len(datasets) + 1):
        for subset in itertools.combinations(datasets, L):
            if len(subset) > 1:
                print(subset)

                sub_list = list(subset)
                for data in sub_list:
                    ensemble_df = pd.DataFrame()
                    for model in sub_list:
                        file_name = path + f'M-' + model + 'D-' + data + '.csv'
                        model_results = pd.read_csv(file_name, sep='\t')
                        if 'en_results' in ensemble_df.columns:
                            combined_result = []
                            for f_val, cur_val in zip(model_results['predictions'], ensemble_df['en_results']):
                                if f_val == 'OFF':
                                    combined_result.append(f_val)
                                else:
                                    combined_result.append(cur_val)
                            ensemble_df['en_results'] = combined_result
                        else:
                            ensemble_df['en_results'] = model_results['predictions']
                    ensemble_df['gold_labels'] = test_dict[data]['labels']
                    macro_f1, micro_f1 = print_information(ensemble_df, 'en_results', 'gold_labels')
                    with open('ensemble_final_results' + path.split('/')[3] + '.txt', 'a') as f:
                        f.write(f'results for : {subset}\n')
                        f.write(f'test set : {data}\n')
                        f.write(f'macro F1 : {macro_f1}\n')
                        f.write('=============================\n\n')
                with open('ensemble_final_results' + path.split('/')[3] + '.txt', 'a') as f:
                    f.write('##################################\n')