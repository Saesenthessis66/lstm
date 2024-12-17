import pandas as pd
import numpy as np
import json 

# Custom DataFrame class to add segment-specific filtering functionality
class SegmentDataFrame(pd.DataFrame):
    # Override the __getitem__ method to allow custom access logic
    def __getitem__(self, key):
        if isinstance(key, str) and key.isdigit():
            # If the key is a string that represents a number, filter rows based on 'Current segment'
            segment_id = float(key)  # Convert key to a float (matches 'Current segment' type)
            return self[self['Current segment'] == segment_id]
        else:
           # Use the default pandas behavior for other keys
            return super().__getitem__(key)
        
# Main class for handling and processing the data
class DataManager:
    _dataFileName: str # Filename of the input CSV file
    _fullData = [] # The full dataset after loading and cleaning
    _allSegments = [] # Unique segment identifiers in the dataset
    _segment_boundaries = {}  # Dictionary of segment start and end boundaries

    # Constructor: Initializes the object and sets up the data
    def __init__(self, dataFile: str):
        self._dataFileName = dataFile  # Store the file name
        self.setUp() # Set up the data by performing initial processing

    def setUp(self):
        self.read_data_from_csv() # Load and clean the data from the CSV file
        self.get_segment_boundaries() # Identify start and end points for each segment
        self.filter_stationary_rows() # Filter rows where vechicle does not move

    # Load data from a CSV file and perform initial cleaning 
    def read_data_from_csv(self):
        # Define the required columns
        required_columns = [
            'X-coordinate', 'Y-coordinate', 'Heading', 
            'Current segment', 'Battery cell voltage'
        ]

        try:
            data = pd.read_csv(self._dataFileName, low_memory=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error reading the file: {e}")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in the file: {', '.join(missing_columns)}")

        # Select only relevant columns and coerce invalid data to NaN
        data = data[required_columns]
        data['X-coordinate'] = pd.to_numeric(data['X-coordinate'], errors='coerce')
        data['Y-coordinate'] = pd.to_numeric(data['Y-coordinate'], errors='coerce')
        data['Heading'] = pd.to_numeric(data['Heading'], errors='coerce')
        data['Battery cell voltage'] = pd.to_numeric(data['Battery cell voltage'], errors='coerce')
        data['Current segment'] = pd.to_numeric(data['Current segment'], errors='coerce')

        # Remove rows with any NaN values
        data = data.dropna()
        
        # Return data as a SegmentDataFrame instead of a regular DataFrame
        self._fullData = SegmentDataFrame(data)

    def filter_stationary_rows(self,  movement_threshold: float = 0.01):
            # Compute differences between consecutive rows
            self._fullData['delta_x'] = self._fullData['X-coordinate'].diff().abs()
            self._fullData['delta_y'] = self._fullData['Y-coordinate'].diff().abs()
            self._fullData['delta_heading'] = self._fullData['Heading'].diff().abs()
            
            # Compute a combined metric for movement
            self._fullData['movement_metric'] = np.sqrt(self._fullData['delta_x']**2 + self._fullData['delta_y']**2) + self._fullData['delta_heading']
            
            # Filter rows where movement is above the threshold
            self._fullData = self._fullData[self._fullData['movement_metric'] > movement_threshold].copy()
            
            # Drop intermediate columns
            self._fullData.drop(columns=['delta_x', 'delta_y', 'delta_heading', 'movement_metric'], inplace=True)

    def get_segment_boundaries(self):
        # Step 1: Trim first and last rows corresponding to incomplete segments
        full_data = self._fullData  # Original dataset

        # Identify the first segment and its consecutive rows
        first_segment = full_data['Current segment'].iloc[0]
        first_segment_indices = full_data[full_data['Current segment'] == first_segment].index
        first_rows_to_drop = first_segment_indices[first_segment_indices <= full_data.index[-1]].tolist()

        # Identify the last segment and its consecutive rows
        last_segment = full_data['Current segment'].iloc[-1]
        last_segment_indices = full_data[full_data['Current segment'] == last_segment].index
        last_rows_to_drop = last_segment_indices[last_segment_indices >= full_data.index[0]].tolist()

        # Trim the data by dropping first and last rows
        trimmed_data = full_data.drop(first_rows_to_drop + last_rows_to_drop)

        # Step 2: Split the trimmed data into consecutive blocks
        segment_blocks = []
        current_segment = None
        current_block = []

        for idx, row in trimmed_data.iterrows():
            segment = row['Current segment']

            if segment != current_segment:
                # Save the current block when the segment changes
                if current_block:
                    segment_blocks.append((current_segment, pd.DataFrame(current_block)))
                # Start a new block
                current_block = [row]
                current_segment = segment
            else:
                current_block.append(row)

        # Save the last block
        if current_block:
            segment_blocks.append((current_segment, pd.DataFrame(current_block)))

        # Step 3: Calculate average start and end points for each segment
        segment_stats = {}
        segment_coordinates = {}

        # Iterate through all segment blocks
        for segment, block in segment_blocks:
            # Get start and end points for the block
            start_x, start_y = block.iloc[0]['X-coordinate'], block.iloc[0]['Y-coordinate']
            end_x, end_y = block.iloc[-1]['X-coordinate'], block.iloc[-1]['Y-coordinate']

            # Accumulate start and end points for each segment
            if segment not in segment_coordinates:
                segment_coordinates[segment] = {
                    'start_x': [], 'start_y': [], 'end_x': [], 'end_y': []
                }
            segment_coordinates[segment]['start_x'].append(start_x)
            segment_coordinates[segment]['start_y'].append(start_y)
            segment_coordinates[segment]['end_x'].append(end_x)
            segment_coordinates[segment]['end_y'].append(end_y)

        # Calculate averages for each segment
        for segment, coords in segment_coordinates.items():
            avg_start_x = sum(coords['start_x']) / len(coords['start_x'])
            avg_start_y = sum(coords['start_y']) / len(coords['start_y'])
            avg_end_x = sum(coords['end_x']) / len(coords['end_x'])
            avg_end_y = sum(coords['end_y']) / len(coords['end_y'])

            segment_stats[segment] = {
                'start_coordinates': [avg_start_x, avg_start_y],
                'end_coordinates': [avg_end_x, avg_end_y]
            }

        # Save the segment boundaries
        self._segment_boundaries = segment_stats
        self.save_segment_boundaries()

    # Save segment boundaries into a .txt file
    def save_segment_boundaries(self):
        try:
            with open('Config/AI/segment_boundaries.txt', 'w') as file: 
                json.dump(self._segment_boundaries, file, indent=4)
            print(f"Segment boundaries successfully written to {'Config/AI/segment_boundaries.txt'}")
        except Exception as e:
            print(f"Error writing segment boundaries to file: {e}")