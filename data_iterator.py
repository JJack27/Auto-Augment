import numpy as np
import PIL
from keras.datasets import cifar10
from transformations import get_transformations

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (x_train, y_train), (x_test, y_test)

# DataIterator class
class DataIterator:
    # shuffle - bool: if the training set is getting shuffled
    # sampling - float: the percentage we sample from training set
    def __init__(self, test=False, batch_size=200, one_hot=True, shuffle=True, sampling=1.0, policy=None):
        (x_train, y_train), (x_test, y_test) = load_data()
        if test:
            self._data = x_test
            self._label = y_test
        else:
            self._data = x_train
            self._label = y_train
        if one_hot:
            one_hot_labels = np.zeros((len(self._label), 10))
            one_hot_labels[np.arange(len(self._label)), self._label.flatten()] = 1
            self._label = one_hot_labels
        self._batch_size = batch_size

        # shuffle before reduce the dataset
        self._num_samples = len(self._data)
        self._num_left = self._num_samples
        self.shuffle = shuffle
        if (shuffle):
            self.shuffle_data()
        
        # reduce dataset
        assert sampling > 0, "Sampling must be greater than 0"
        assert sampling <= 1, "Sampling must be less than 0"

        self._data = self._data[: int(len(self._data) * sampling)]
        self._label = self._label[: int(len(self._label) * sampling)]

        self._num_samples = len(self._data)
        self._num_left = self._num_samples
        self._batch_pointer = 0

        

        if policy == None:
            #raise Exception("Policy not given. Need to provide policy")
            print("No policy given. Using dataset without data augmentation.")
        else:
            self._policy = policy
            
            # Note that for pairing sample, the img2 is chose randomly from the entire dataset
            self._transformations = get_transformations(x_train)
            self._apply_policy()

        


    def shuffle_data(self):
        np.random.seed(1)
        image_indices = np.random.permutation(np.arange(self._num_samples))
        self._data = self._data[image_indices]
        self._label = self._label[image_indices]

    def next_batch(self):
        if (self._batch_size <= self._num_left):
            batch_x = self._data[self._batch_pointer : self._batch_pointer + self._batch_size]
            batch_y = self._label[self._batch_pointer : self._batch_pointer + self._batch_size]
            self._batch_pointer += self._batch_size
        elif (self._num_left != 0 and self._num_left < self._batch_size):
            batch_x_1 = self._data[self._batch_pointer :]
            batch_y_1 = self._label[self._batch_pointer :]
            if (self.shuffle):
                self.shuffle_data()
            batch_x_2 = self._data[0: self._batch_size - self._num_left]
            batch_y_2 = self._label[0: self._batch_size - self._num_left]

            batch_x = np.vstack((batch_x_1, batch_x_2))
            batch_y = np.vstack((batch_y_1, batch_y_2))
            self._num_left = 0
        else:
            batch_x = None
            batch_y = None
        self._num_left -= self._batch_size  

        return batch_x, batch_y

    @property
    def data_and_label(self):
        return (self._data, self._label)

    '''
    Apply policies to original dataset, and append modified images to the end
    Shuffle the dataset is self._shuffle is True
    '''
    def _apply_policy(self):
        i = 0
        num_batches = 0
        augmented_data = []
        augmented_label = []
        # loop over batches
        while(i < self._num_samples):
            # randomly select policy for each batch
            policy = np.random.choice(len(self._policy)//2)
            policy = [self._policy[policy*2], self._policy[policy*2 + 1]]
            num_batches += 1
            # get mini-batch
            if (self._batch_size <= self._num_left):
                batch_x = self._data[i : i + self._batch_size].copy()
                batch_y = self._label[i : i + self._batch_size].copy()
                i += self._batch_size
                self._num_left -= self._batch_size
            elif (self._num_left != 0 and self._num_left < self._batch_size):
                batch_x = self._data[i :].copy()
                batch_y = self._label[i :].copy()
                i, self._num_left = self._num_samples
            
            for p in policy:
                # calculating magnitudes for each policy
                v = (p[2]+1) / 10
                v *= (self._transformations[p[0]][2] - self._transformations[p[0]][1])
                v += self._transformations[p[0]][1]

                # if random is less than generated probability
                if np.random.random() < p[1] / 10:
                    batch = []
                    for x in range(len(batch_x)):
                        x = PIL.Image.fromarray(batch_x[x])
                        x = self._transformations[p[0]][0](x, v)
                        batch.append(np.array(x))
                    batch_x = np.array(batch)
            augmented_data.append(np.array(batch_x))
            augmented_label.append(np.array(batch_y))
        
        augmented_data = np.array(augmented_data)
        augmented_data = np.reshape(augmented_data, [-1, 32, 32, 3])
        augmented_label = np.array(augmented_label)
        augmented_label = np.reshape(augmented_data, [-1, 10])
        self._data = np.append(self._data, np.array(augmented_data), axis=0) 
        self._label = np.append(self._label, np.array(augmented_label),axis=0) 

        # update num_samples 
        self._num_samples *= 2
        self._num_left = self._num_samples
        
        self.shuffle_data()


