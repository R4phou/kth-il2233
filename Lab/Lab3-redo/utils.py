import numpy as np
import matplotlib.pyplot as plt

def read_tsv(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [line.strip().split('\t') for line in data]
    data = np.array(data)
    return data


def remove_labels_from_data(data):
    # labels are the first column
    labels = data[:, 0]
    data = data[:, 1:]
    return data, labels

BME_data = read_tsv('./dataset/BME_TRAIN.tsv')
BME_data, BME_labels = remove_labels_from_data(BME_data)

def transform_into_c_array(data):
    """
    return 
    double BME[n]
    """
    nb_features = data.shape[1]
    nb_samples = data.shape[0]
    c_array = f'double BME[{nb_features*nb_samples}]'
    c_array += ' = {'                    
    for i in range(nb_samples):
        for j in range(nb_features):
            c_array += f'{data[i, j]}, '
    c_array = c_array[:-2] + '};'
    return c_array                               

def print_array_in_file(c_array, file_path):
    with open(file_path, 'w') as f:
        f.write(c_array)
    
c_array = transform_into_c_array(BME_data)
print_array_in_file(c_array, './dataset/BME_TRAIN.h')

