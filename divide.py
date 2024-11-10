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

def read_data_from_csv():
    data = pd.read_csv('agv.pkl', low_memory=False)
    data = data[['Timestamp', 'X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']]
    data['X-coordinate'] = pd.to_numeric(data['X-coordinate'], errors='coerce')
    data['Y-coordinate'] = pd.to_numeric(data['Y-coordinate'], errors='coerce')
    data['Heading'] = pd.to_numeric(data['Heading'], errors='coerce')
    data['Current segment'] = pd.to_numeric(data['Current segment'], errors='coerce')
    data = data.dropna()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.index = data.pop('Timestamp')
    
    # Return data as a SegmentDataFrame instead of a regular DataFrame
    return SegmentDataFrame(data)
    
def create_interpolated_dataframes(data, segments):

    # Initialize array that stores data for all segments
    final_data = []

    for i in range(0,len(segments)):

        # Copy unaltered data from file
        df = data.copy()

        # Assign current segment from an array
        X = segments[i]

        # Filter data to segment X
        df = df[df['Current segment'] == X]

        # Remove rows where Current segment is 56 and Y-coordinate > 18
        if X == 56.0:
            df = df[~(df['Y-coordinate'] > 18)]

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

    return final_data

def find_close_segments(segments_data, segments, distance_threshold=1.0):

    close_pairs = []
    
    # Extract the last point of each segment (endpoint) and the first point (start point)
    segment_endpoints = [(segment.iloc[-1], segment.iloc[0]) for segment in segments_data]

    # Iterate over each pair of segments
    for i, (end_i, _) in enumerate(segment_endpoints):
        for j, (_, start_j) in enumerate(segment_endpoints):
            # Skip if the indices are the same
            if i == j:
                continue
            
            # Calculate Euclidean distance between end of segment i and start of segment j
            distance = np.sqrt((end_i['X-coordinate'] - start_j['X-coordinate'])**2 +
                               (end_i['Y-coordinate'] - start_j['Y-coordinate'])**2)
            
            # Check if distance is within the threshold
            if distance <= distance_threshold:
                # Use the actual segment ID from `segments` list
                close_pairs.append((segments[i], segments[j]))  # Store segment IDs instead of indices
    
    return close_pairs

def get_unique_segments(data):
    # Extract unique segments from the 'Current segment' column
    unique_segments = data['Current segment'].unique()
    return unique_segments

def create_segment_dictionary(segment_ids, dataframes):

    # Create a dictionary by zipping segment_ids and dataframes
    segment_dict = {str(segment_id): df for segment_id, df in zip(segment_ids, dataframes)}
    return segment_dict

def create_scaler(data):
    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    columns_to_scale = ['X-coordinate', 'Y-coordinate', 'Heading']
    scaler.fit(data[columns_to_scale])  # Fit only on the specified columns

    # Save the scaler to disk
    joblib.dump(scaler, 'scaler.pkl')


# data = read_data_from_csv()

# segments = get_unique_segments(data)

# # Generate the segment data
# segmented_df = create_interpolated_dataframes(data, segments)

# # Find close segments using the specified distance threshold
# close_segments = find_close_segments(segmented_df, segments, distance_threshold=1.5)

# print("Close segment pairs:", close_segments)

# segment_dict = create_segment_dictionary(segments, segmented_df)

# # Accessing the DataFrame for segment 15:
# print(segment_dict['59.0'])