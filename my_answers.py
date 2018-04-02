import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import keras


# The function below transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # looping for P - T times where P is the length of series and T is the window size
    for it in range(len(series) - window_size):
        X.append(series[it : it + window_size])
        y.append(series[it + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    # building the required model
    model = Sequential()
    model.add(LSTM(units = 5, input_shape = (window_size, 1)))
    model.add(Dense(units = 1))

    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    atypical = [] # list to store all atypical characters
    for t in text:
        if (not t.isalpha()) and (t not in punctuation) and (t not in atypical):
            atypical.append(t)

    # replacing the atypical characters with space character
    for a in atypical:
        text = text.replace(a, ' ')
    return text


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # similar loop from before which now includes the steps by which to shift the iterator
    for it in range(0, len(text) - window_size, step_size):
        inputs.append(text[it: it + window_size])
        outputs.append(text[it + window_size])

    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    
    # building the required model
    model = Sequential()
    model.add(LSTM(units = 200, input_shape = (window_size, num_chars)))
    model.add(Dense(units = num_chars))
    model.add(Activation('softmax'))

    return model
