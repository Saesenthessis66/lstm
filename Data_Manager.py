import pandas as pd
import numpy as np
import json 

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
    _segment_boundaries = {}

    def __init__(self, dataFile: str):
        self._dataFileName = dataFile
        self.setUp()

    def setUp(self):
        self.read_data_from_csv()
        self.get_segment_boundaries()
        self.find_jumps(3)
        self._fullData.drop(columns=['Timestamp'],inplace=True)
        self._divided_data = sorted(self._divided_data, key=len)

        
    def read_data_from_csv(self):
        data = pd.read_csv(self._dataFileName, low_memory=False)
        data = data[['Timestamp', 'X-coordinate', 'Y-coordinate', 'Heading', 'Current segment','Going to ID']]
        data['X-coordinate'] = pd.to_numeric(data['X-coordinate'], errors='coerce')
        data['Y-coordinate'] = pd.to_numeric(data['Y-coordinate'], errors='coerce')
        data['Heading'] = pd.to_numeric(data['Heading'], errors='coerce')
        data['Going to ID'] = pd.to_numeric(data['Going to ID'], errors='coerce')
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

    def get_segment_boundaries(self):
        segments = self._fullData['Current segment'].unique()  # Get unique segment IDs
        self._allSegments = segments
        segment_boundaries = {}

        for segment in segments:
            # Filter rows belonging to the current segment
            segment_data = self._fullData[self._fullData['Current segment'] == segment]

            # Extract start and end points
            start_index = segment_data.index[0]
            end_index = segment_data.index[-1]
            start_point = segment_data.loc[start_index, ['X-coordinate', 'Y-coordinate']].tolist()
            end_point = segment_data.loc[end_index, ['X-coordinate', 'Y-coordinate']].tolist()

            # Store results in the dictionary
            segment_boundaries[segment] = {
                'start_coordinates': start_point,
                'end_coordinates': end_point
            }

        self._segment_boundaries =  segment_boundaries
        self.save_segment_boundaries()

    def save_segment_boundaries(self):
        try:
            with open('segment_boundaries.txt', 'w') as file:
                json.dump(self._segment_boundaries, file, indent=4)
            print(f"Segment boundaries successfully written to {'segment_boundaries.txt'}")
        except Exception as e:
            print(f"Error writing segment boundaries to file: {e}")