import os
import numpy as np

# Path for exported data
DATA_PATH = os.path.join('MP_Data')

# words/ phrases for detection
actions = np.array(['hello', 'thanks', 'hehe hi sprout'])

# thirty videos of data (wtf)
number_of_sequences = 30

# videos are 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(number_of_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            # makedirs, meaning makes folder with the name of the current action in actions

            # hello
            ## folder for video 0, sequence_length frames
            ## folder for video 1, sequence_length frames
            ## ..
            ## folder for number_of_sequences - 1, sequence_length frames
            # thanks
            ## ..
            #.. etc
        except:
            pass