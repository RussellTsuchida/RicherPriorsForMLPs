import numpy as np
import os.path

class ExperimentArray(object):
    """
    A 2D array that persists on disk across multiple processes. Each time an 
    array index is updated, it is stored as a temporary file on disk. The 
    entire array can then be read by reading each array index from disk and
    assembling it into a numpy array.
    """
    def __init__(self, size, dirname):
        """
        Args:
            size (list): size of the array
            dirname (str): directory to store array data in
        """
        os.makedirs(dirname, exist_ok=True)
        self.dirname = dirname
        self.shape = size

    def __setitem__(self, index, value):
        """
        Save the value at index to disk
        
        Args:
            index (tuple): row and column index of data to set
            value (float): float value to assign at index
        """
        print(value)
        np.save(self.dirname + 'temp' + \
                str(index[0]) + '_' + str(index[1]), value)
        
    def __getitem__(self, index):
        """
        Load the value at aindex from disk

        Args:
            index (tuple): row and column index of data to get
        """
        fname = self.dirname + 'temp' + \
            str(index[0]) + '_' + str(index[1]) + '.npy'
        # Check if file exists, if it does return that value
        if os.path.isfile(fname):
            return np.load(self.dirname + 'temp' + \
                str(index[0]) + '_' + str(index[1]) + '.npy')
        else:
            # If not, default to zero
            return 0

    def load_array(self):
        """
        Read all the data from disk and return it as a single array.
        """
        data = np.empty(self.shape)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                data[row, col] = self.__getitem__((row,col))

        return data

if __name__ == '__main__':
    DIRNAME = 'code/experiments/outputs/test/'
    arr1 = ExperimentArray((3,3), DIRNAME)
    arr2 = ExperimentArray((3,3), DIRNAME)
    arr1[0,2] = 5.7
    arr2[2,1] = -6

    data = arr1.load_array()
    print(data)
    print(data[0,:])

