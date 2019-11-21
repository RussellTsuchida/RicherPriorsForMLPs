import numpy as np
import os.path
import pandas as pd

class Dataset(object):
    """
    An abstract class for a dataset.
    """
    def __init__(self, out_dir, name='dataset'):
        """
        Args:
            out_dir (dir): directory to save and load data to and from.
            name (str): name for dataset.
        """
        self.out_dir = out_dir
        self.name = name

    def _generate_data(self):
        raise NotImplementedError
    
    def _save_data(self, X_train, Y_train, X_test, Y_test, save=True):
        if save:
            np.save(self.out_dir + self.name + "X_train.npy", X_train, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "Y_train.npy", Y_train, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "X_test.npy",  X_test, 
                    allow_pickle=False)
            np.save(self.out_dir + self.name + "Y_test.npy",  Y_test, 
                    allow_pickle=False)

    def _load_data(self):
        X_train = np.load(self.out_dir + self.name + 'X_train.npy')
        Y_train = np.load(self.out_dir + self.name + 'Y_train.npy')
        X_test  = np.load(self.out_dir + self.name + 'X_test.npy')
        Y_test  = np.load(self.out_dir + self.name + 'Y_test.npy')

        return [X_train, Y_train, X_test, Y_test]

    def load_or_generate_data(self, force_generate=False, save=True):
        files_exist =   os.path.isfile(self.out_dir + self.name + \
                                "X_train.npy") and\
                        os.path.isfile(self.out_dir + self.name + \
                                "Y_train.npy") and\
                        os.path.isfile(self.out_dir + self.name + \
                        "X_test.npy")

        if (force_generate) or (not files_exist):
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                    self._generate_data()
            self._save_data(\
                    self.X_train, self.Y_train, self.X_test, self.Y_test, save)

        else:
            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                    self._load_data()

        return [self.X_train, self.Y_train, self.X_test, self.Y_test]

class Mauna(Dataset):
    """
    https://iridl.ldeo.columbia.edu/SOURCES/.KEELING/.MAUNA_LOA/co2/T+exch+table-+text+text+skipanyNaN+-table+.html
    """
    def __init__(self, out_dir, train_size=100, test_size=349):
        self.train_size = train_size
        self.test_size = test_size
        super().__init__(out_dir = out_dir, name='mauna')

    def _generate_data(self):
        fname = self.out_dir + 'mauna.csv'
        data = np.loadtxt(fname, delimiter=',', usecols=(4,5))

        half_train_size = int(self.train_size/2)
        X_train = np.hstack(   (data[:half_train_size, 0], 
                                data[-half_train_size:, 0])).reshape((-1,1))
        Y_train = np.hstack(   (data[:half_train_size, 1], 
                                data[-half_train_size:, 1])).reshape((-1,1))
        X_test = data[half_train_size:-half_train_size,0].reshape((-1,1))
        Y_test = data[half_train_size:-half_train_size,1].reshape((-1,1))

        return [X_train, Y_train, X_test, Y_test]

class Yacht(Dataset):
    """
    http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics#
    """
    def __init__(self, out_dir, train_size=100, test_size=208):
        assert (train_size + test_size) == 308
        self.train_size = train_size
        self.test_size = test_size
        super().__init__(out_dir, name='yacht')

    def _generate_data(self):
        fname = self.out_dir + 'yacht.csv'
        data = np.loadtxt(fname, delimiter=' ')
        np.random.shuffle(data)

        X_train = data[:self.train_size, :6]
        Y_train = data[:self.train_size, 6].reshape((-1,1))
        X_test = data[self.train_size:, :6]
        Y_test = data[self.train_size:, 6].reshape((-1, 1))
        
        return [X_train, Y_train, X_test, Y_test]
    
class Ghcn(Dataset):
    """

    """
    def __init__(self, out_dir, lat_bot=-90,
            lat_top = 90, long_left=-180, long_right=180, train_years=None,
            test_years = None):
        """
        https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/

        The idea is to try and get a model to interpolate between years.
        E.g.    train_years = [2000, 2001, 2002,            2005, 2006, 2007]
                test_years  = [                 2003, 2004                  ]

        Args:
            lat_bot (float): Consider all weather stations above this latitude
                and below lat_top.
            lat_top (float): Consider all weather stations below this latitude 
                and above lat_bot.
            long_left (float): Consider all weather stations to the right of 
                this longitude and to the left of long_right.
            long_right (float): Consider all weather stations to the left of 
                this latitude and to the right of long_left.
        """
        self.lat_bot = lat_bot
        self.lat_top = lat_top
        self.long_left = long_left
        self.long_right = long_right
        self._set_years(train_years, test_years)
        super().__init__(out_dir = out_dir, name='GHCN')

    def _set_years(self, train_years, test_years):
        if train_years is None:
            train_years = list(range(1701, 2020))
        if test_years is None:
            test_years = []
        self.train_years = train_years
        self.test_years = test_years
           
    def _get_df_from_dat(self, fname_dat, fname_meta):
        """

        Returns:
            pandas df containing the columns:
            
            | time | year | month | lat | long | elev | station | value |
        """
        path = os.path.abspath(os.path.dirname(__file__))
        path_data = os.path.join(path, fname_dat)
        path_meta = os.path.join(path, fname_meta)

        # Read each column of the data. Station/year and months.
        self.months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                'sep','oct', 'nov', 'dec']
        colspecs = [(19, 24)]
        for i, month in enumerate(self.months[1:]):
            old_interval = colspecs[i]
            start = old_interval[1] + 3
            colspecs = colspecs + [(start, start+5)]
        df = pd.read_fwf(path_data, header=None,
                colspecs=[(0,19) ] + colspecs,
                names=['mixed'] + self.months)

        # Split first column into station and year
        df['station'] = df['mixed'].map(lambda x: x[0:11])
        df['year'] = df['mixed'].map(lambda x: x[11:15])

        # Load the information about the stations from the metadata.
        meta_df = pd.read_csv(path_meta, header=None,
                        names=['station', 'lat', 'long', 'elev', 'name'],
                        delim_whitespace=True)
        lat_dict = meta_df.set_index('station').to_dict()['lat']
        long_dict = meta_df.set_index('station').to_dict()['long']
        elev_dict = meta_df.set_index('station').to_dict()['elev']

        # Note, some info about stations is missing e.g. CA003023740
        # remove these points
        df = df[df['station'].isin(list(lat_dict.keys()))]
        # Get latitude, longitude and elevation for each station, as well as year
        df['lat'] = df['station'].map(lambda x: lat_dict[x])
        df['long'] = df['station'].map(lambda x: long_dict[x])
        df['elev'] = df['station'].map(lambda x: elev_dict[x])
        df[['year', 'lat', 'long', 'elev']+ self.months] = \
        df[['year', 'lat', 'long', 'elev'] + self.months].apply(pd.to_numeric)

        # Convert month columns to rows using melt
        df = pd.melt(df, id_vars=['year', 'station', 'lat', 'long', 'elev'],
                value_vars=self.months)
        # Note: only January
        #df = df[df['variable'].isin(['jan', 'jul'])]
        
        # The new time column represents time as a float, with each month 
        # being of length 1/12
        df['time'] = df['year'] + df['variable'].\
                map(lambda x: self.months.index(x)/12.)

        return df
   
    def _generate_data(self):
        """
        https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/
        """
        df = self._get_df_from_dat('ghcnm.tavg.v4.0.1.20190911.qcu.dat',
                'ghcnm.tavg.v4.0.1.20190911.qcu.inv')

        # Remove missing entries which are populated with -9999
        df = df[~df.eq(-9999).any(1)]
        """
        Temperature values are in
        hundredths of a degree Celsius, but are expressed as whole
        integers (e.g. divide by 100.0 to get whole degrees Celsius).
        """
        df['value'] = df['value'] / 100.

        df = self._filter_lat_long_range(df, self.lat_bot, self.lat_top,
                self.long_left, self.long_right)

        #df = df.apply(pd.to_numeric)
        #df = df.convert_objects(convert_numeric=True)
        # Only get the values specified by train_years

        self.df_train = df.copy()[df['year'].isin(self.train_years)]
        self.df_test = df.copy()[df['year'].isin(self.test_years)]

        # Remove fields that we won't use
        self.df_train = self.df_train.drop(['year', 'station', 'variable'], axis=1)
        self.df_test = self.df_test.drop(['year', 'station', 'variable'], axis=1)

        self.df_train = self.df_train[['value', 'lat', 'long', 'elev', 'time']]
        self.df_test = self.df_test[['value', 'lat', 'long', 'elev', 'time']]

        X_train, Y_train = self.get_numpy_train_data()
        X_test, Y_test = self.get_numpy_star_data()

        return [X_train, Y_train, X_test, Y_test]      

    def get_numpy_train_data(self):
        data = self.df_train.values
        # Data columns are ordered [lat, long, elev, value, time]

        X_train = data[:,1:]
        Y_train = np.reshape(data[:,0], (-1,1))

        return [X_train, Y_train]

    def get_numpy_star_data(self):
        data = self.df_test.values
        X_test = data[:,1:]
        Y_test = np.reshape(data[:,0], (-1,1))

        return [X_test, Y_test]

    def _filter_lat_long_range(self, df, lat_bot, lat_top, long_left, 
            long_right):
        df = df[(df['lat'] > lat_bot)]
        df = df[(df['lat'] < lat_top)]
        df = df[(df['long'] > long_left)]
        df = df[(df['long'] < long_right)]
        return df

    def generate_lat_long_grid(self, time, elev):
        """
        Generate a latitude/longitude grid over the region specified by 
        internal attributes at a given time and elevation. Returned array
        has columns.
        | lat | long | elev | year |
        """
        lats = np.linspace(self.lat_bot, self.lat_top, 100)
        longs = np.linspace(self.long_left, self.long_right, 100)
        
        # Cartesian product between lats and longs
        X = np.transpose([np.tile(lats, len(longs)), np.repeat(longs, len(lats))])

        # Add time and elevation columns
        X = np.hstack(\
            (X, np.ones((X.shape[0], 1))*elev, np.ones((X.shape[0], 1))*time))

        return X

class Normaliser(object):
    """
    Normalise and unnormalise data. The following transformation is applied:

    X <- (X - a)/ b

    If mode is standardise, a and b are the mean and standard deviations of the
    columns of X. If mode is normalise, a and b are the minimum and range of 
    columns of X.
    """
    def __init__(self, X, Y, mode = 'standardise'):
        """
        Args:
            X (nparray): (n, d_in) array of raw data inputs.
            Y (nparray): (n, d_out) array of raw data targets.
            mode (str): type of normalisation to perform.
        """
        self.X = X
        self.Y = Y
        
        if mode == 'standardise':
            self.Xa = np.mean(X, axis=0)
            self.Ya = np.mean(Y, axis=0)
            self.Xb = np.std(X, axis=0)
            self.Yb = np.std(Y, axis=0)
        elif mode == 'normalise':
            self.Xa = np.amin(X, axis=0)
            self.Ya = np.amin(Y, axis=0)
            self.Xb = np.amax(X, axis=0) - np.amin(X, axis=0)
            self.Yb = np.amax(Y, axis=0) - np.amin(Y, axis=0)
        else:
            raise NotImplementedError

    def normalised_XY(self, X = None, Y=None):
        """
        Returns:
            Normalised copy of [X, Y]
        """
        if (X is None) or (Y is None):
            assert (X is None) and (Y is None)
            X = self.X
            Y = self.Y

        X = (X - self.Xa)/ self.Xb
        Y = (Y - self.Ya)/ self.Yb

        return [X, Y]

    def unnormalised_XY(self, X = None, Y = None):
        """
        Returns:
            Un-normalised copy of [X, Y]
        """
        if (X is None) or (Y is None):
            assert (X is None) and (Y is None)
            return [self.X, self.Y]

        X = X*self.Xb + self.Xa
        Y = Y*self.Yb + self.Ya

        return [X, Y]
 
    def normalised_X(self, X = None):
        if X is None:
            X = self.X
        return ( X - self.Xa ) / self.Xb

    def unnormalised_Y(self, Y = None):
        if Y is None:
            Y = self.Y
        return Y*self.Yb + self.Ya



