import pandas as pd
import numpy as np

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
    _allSegments = []
    _average_points_amount = {}
    _day_divided_data = []
    _divided_data = []

    def __init__(self, dataFile: str):
        self._dataFileName = dataFile
        self.setUp()

    def setUp(self):
        self.read_data_from_csv()
        self.get_unique_segments()
        self.count_average_point_amount()
        self.find_jumps(3)
        self._fullData.drop(columns=['Timestamp'],inplace=True)

        
    def read_data_from_csv(self):
        data = pd.read_csv(self._dataFileName, low_memory=False)
        data = data[['Timestamp', 'X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']]
        data['X-coordinate'] = pd.to_numeric(data['X-coordinate'], errors='coerce')
        data['Y-coordinate'] = pd.to_numeric(data['Y-coordinate'], errors='coerce')
        data['Heading'] = pd.to_numeric(data['Heading'], errors='coerce')
        data['Current segment'] = pd.to_numeric(data['Current segment'], errors='coerce')
        data = data.dropna()
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Return data as a SegmentDataFrame instead of a regular DataFrame
        self._fullData = SegmentDataFrame(data)


    def find_jumps(self, threshold):
        segments = []
        last_index = 0  # Start from the beginning of the dataset
        
        for i in range(1, len(self._fullData)):
            x1, y1 = self._fullData.iloc[i-1][['X-coordinate', 'Y-coordinate']]
            x2, y2 = self._fullData.iloc[i][['X-coordinate', 'Y-coordinate']]
            
            # Calculate Euclidean distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance > threshold:
                # Append the segment from last_index to the current point
                segments.append(self._fullData.iloc[last_index:i])
                last_index = i  # Update the last index to the current point
        
        # Append the remaining data after the last jump
        if last_index < len(self._fullData):
            segments.append(self._fullData.iloc[last_index:])
        
        self._divided_data = segments

    def count_average_point_amount(self):
        # Initialize a dictionary to store segment averages
        avg_values = {}

        # Loop through all segments
        for i in range(0, len(self._allSegments)):
            # Copy unaltered data from file
            df = self._fullData.copy()

            # Assign current segment from an array
            X = self._allSegments[i]

            # Filter data to segment X
            df = df[df['Current segment'] == X]

            # Check if the segment data is empty
            if df.empty:
                avg_values[X] = 0  # Set average to 0 if no data
                continue

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
            avg_len = len(df) / len(dataframes) if dataframes else 0

            # Store the result in the dictionary with the segment as the key
            avg_values[X] = int(avg_len)

        self._average_points_amount = avg_values

    def get_unique_segments(self):
        # Extract unique segments from the 'Current segment' column
        self._allSegments = self._fullData['Current segment'].unique()