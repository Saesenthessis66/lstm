import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

class SegmentDataFrame(pd.DataFrame):
    # Override __getitem__ for custom behavior
    def __getitem__(self, key):
        if isinstance(key, str) and key.isdigit():
            # If key is a string representation of a number, filter by 'Current segment'
            segment_id = float(key)  # Convert to float, assuming 'Current segment' is a float
            return self[self['Current segment'] == segment_id]
        else:
            # Use regular indexing behavior for other keys
            return super().__getitem__(key)
        
class DataDivision:
    _dataFileName: str = ''
    _fullData = []
    _interpolatedData = []
    _closePairs = []
    _dictionaryData = []
    _allSegments = []
    _scaler = any

    def __init__(self, dataFile: str):
        self._dataFileName = dataFile
        self.setUp()

    def setUp(self):
        self.read_data_from_csv()
        self.create_scaler()
        self.get_unique_segments()
        self.create_interpolated_dataframes()
        self.create_segment_dictionary()
        self.find_close_segments()
        
    def read_data_from_csv(self):
        data = pd.read_csv(self._dataFileName, low_memory=False)
        data = data[['X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']]
        data['X-coordinate'] = pd.to_numeric(data['X-coordinate'], errors='coerce')
        data['Y-coordinate'] = pd.to_numeric(data['Y-coordinate'], errors='coerce')
        data['Heading'] = pd.to_numeric(data['Heading'], errors='coerce')
        data['Current segment'] = pd.to_numeric(data['Current segment'], errors='coerce')
        data = data.dropna()
        
        # Return data as a SegmentDataFrame instead of a regular DataFrame
        self._fullData = SegmentDataFrame(data)
    
    def create_interpolated_dataframes(self):

        # Initialize array that stores data for all segments
        final_data = []

        for i in range(0,len(self._allSegments)):

            # Copy unaltered data from file
            df = self._fullData.copy()

            # Assign current segment from an array
            X = self._allSegments[i]

            # Filter data to segment X
            df = df[df['Current segment'] == X]

            # Initialize variable for computing average dataframe length  
            avg_len = len(df)

            # Set distance for max distance between two consecutive points
            distance_threshold = 1

            # Calculate Euclidean distance explicitly between each pair of consecutive points
            df['Distance'] = np.sqrt(
                (df['X-coordinate'] - df['X-coordinate'].shift(1))**2 + 
                (df['Y-coordinate'] - df['Y-coordinate'].shift(1))**2
            )

            # Initialize list for segmented dataframes
            dataframes = []
            start_idx = 0

            # Loop to split the dataframe based on the distance threshold
            for idx in range(1, len(df)):
                if df['Distance'].iloc[idx] > distance_threshold:
                    segment_df = df.iloc[start_idx:idx].copy()  # Create a new segment
                    dataframes.append(segment_df)               # Append segment to list
                    start_idx = idx                             # Update start index for the next segment

            # Append the last segment if any rows remain
            if start_idx < len(df):
                segment_df = df.iloc[start_idx:].copy()
                dataframes.append(segment_df)

            # Optionally remove the Distance column from each segment if no longer needed
            for segment in dataframes:
                segment.drop(columns=['Distance'], inplace=True)

            # Compute average amount of points in one run in segment                      
            avg_len /= len(dataframes) 

            # Number of points to resample (you can adjust this number)
            num_points = 100

            # Interpolate each dataframe to have the same number of points
            interpolated_dfs = []
            for df in dataframes:
                # Create a fixed range of values from 0 to 1 to act as a new "index"
                resample_index = np.linspace(0, 1, num_points)
                # Interpolate X and Y coordinates using this new index
                df_interpolated = pd.DataFrame({
                    'X-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['X-coordinate']),
                    'Y-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['Y-coordinate']),
                    'Heading': np.interp(resample_index, np.linspace(0, 1, len(df)), df['Heading'])
                })
                interpolated_dfs.append(df_interpolated)

            # Concatenate and calculate the average for each coordinate across all dataframes
            average_df = pd.concat(interpolated_dfs).groupby(level=0).mean()

            # Append averaged data for segment in array
            final_data.append(average_df)

        self._interpolatedData = final_data 

    def get_unique_segments(self):
        # Extract unique segments from the 'Current segment' column
        self._allSegments = self._fullData['Current segment'].unique()

    def create_segment_dictionary(self):

        # Create a dictionary by zipping segment_ids and dataframes
        segment_dict = {str(segment_id): df for segment_id, df in zip(self._allSegments, self._interpolatedData)}
        self._dictionaryData = segment_dict

    def create_scaler(self):
        # Initialize and fit the scaler
        scaler = MinMaxScaler()
        columns_to_scale = ['X-coordinate', 'Y-coordinate', 'Heading']
        scaler.fit(self._fullData[columns_to_scale])  # Fit only on the specified columns

        # Save the scaler to disc
        joblib.dump(scaler, 'scaler.pkl')
        self._scaler = scaler

    def find_close_segments(self):
        close_segments = set()  # Use a set to avoid duplicates
        prev_segment = None  # Track the previous segment

        for idx, row in self._fullData.iterrows():
            current_segment = row["Current segment"]
            
            # Check if the segment changes
            if prev_segment is not None and current_segment != prev_segment:
                # Add the pair of segments in the original order
                close_segments.add((prev_segment, current_segment))
            
            # Update previous segment
            prev_segment = current_segment

        self._closePairs = close_segments
