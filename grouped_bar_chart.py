import argparse

from visualize import compare_best

from collections import OrderedDict

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('result_files', nargs='+', help="Path to results.txt file")
    ap.add_argument('--dataset_name', help="Name of dataset", default="DATASETNAME")
    args = ap.parse_args()
    
    dataset_metadata_mse, dataset_metadata_mae = {}, {}
    
    for fl in args.result_files:
        key = int(fl.split('_')[-1][:-4])
        
        with open(fl, 'r') as f:
            lines = list(f.readlines())

        for line in lines:
            if "START=" in line:
                params = line.split(': ')[-1]
        
        index = 0
        for line in lines:
            if "Min:" in line:
                metric = float(line.split('Min: ')[-1])
                
                if index == 0:
                    lst = [1,1,metric,0.5]

                    dataset_metadata_mse[key] = [lst]
                
                elif index == 1:
                    lst = [1,1,metric,0.5]
                
                    dataset_metadata_mae[key] = [lst]
                    
                elif index == 2:
                    lst = [float(x.split('=')[-1]) for x in params.split(' ')]

                    lst[-1] = lst[-2]
                    lst[-2] = metric

                    dataset_metadata_mse[key].append(lst)

                elif index == 3:
                    lst = [float(x.split('=')[-1]) for x in params.split(' ')]

                    lst[-1] = lst[-2]
                    lst[-2] = metric
                
                    dataset_metadata_mae[key].append(lst)
                    
                index+=1

    
    dataset_metadata_mse = OrderedDict(dataset_metadata_mse)
    dataset_metadata_mae = OrderedDict(dataset_metadata_mae)
    compare_best(dataset_metadata_mse, args.dataset_name, mse=True)
    compare_best(dataset_metadata_mae, args.dataset_name, mse=False)
