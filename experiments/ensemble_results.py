import itertools

datasets = ['hasoc', 'hatexplain', 'olid','davidson']
combinations = set()

for L in range(len(datasets) + 1):
    if L > 1:
        for subset in itertools.combinations(datasets, L):
            print(subset)

        # filename = "ensemble_results/" + "M-" + model + "D-" + dataset + ".csv"




