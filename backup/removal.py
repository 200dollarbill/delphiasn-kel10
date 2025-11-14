import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# The 'os' and 'csv' imports are no longer needed

# --- create_sample_csv function has been removed ---

def clean_sensor_data(df, max_val=4096):
    """
    Cleans the sensor data by interpolating values exceeding max_val.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'value' column.
        max_val (int): The maximum allowed sensor value.
        
    Returns:
        pd.DataFrame: A new DataFrame with cleaned data.
    """
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # 1. Mark all data points exceeding max_val as 'NaN' (Not a Number)
    #    This flags them as missing data.
    cleaned_df.loc[cleaned_df['value'] > max_val, 'value'] = np.nan
    
    # 2. Use linear interpolation to fill the 'NaN' gaps.
    #    This calculates a new value for each gap based on a straight line
    #    between the nearest *valid* data points (before and after).
    #    'limit_direction="both"' handles NaNs at the start and end of the file.
    cleaned_df['value'] = cleaned_df['value'].interpolate(
        method='linear', 
        limit_direction='both'
    )
    
    return cleaned_df

def plot_comparison(original_df, cleaned_df, max_val=4096):
    """
    Plots the original and cleaned data side-by-side.
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Original Data ---
    # Note: We use 'timestamp' and 'value' as the column names
    # after renaming them in the main() function.
    ax1.plot(original_df['timestamp'], original_df['value'], 
             'b-', label='Original Data')
    
    # Highlight the corrupt data points
    corrupt_data = original_df[original_df['value'] > max_val]
    ax1.plot(corrupt_data['timestamp'], corrupt_data['value'], 
             'ro', label='Corrupt Data (> 4096)')
             
    # Show the 4096 limit
    ax1.axhline(y=max_val, color='r', linestyle='--', 
                label=f'Max Limit ({max_val})')
    
    ax1.set_title('Before Cleaning')
    ax1.set_ylabel('Sensor Value')
    ax1.legend()
    ax1.grid(True, linestyle=':')
    
    # --- Plot 2: Cleaned Data ---
    ax2.plot(cleaned_df['timestamp'], cleaned_df['value'], 
             'g-', label='Cleaned Data (Interpolated)')
             
    # Show the 4096 limit
    ax2.axhline(y=max_val, color='r', linestyle='--', 
                label=f'Max Limit ({max_val})')
    
    ax2.set_title('After Cleaning')
    ax2.set_xlabel('Timestamp (from Index column)')
    ax2.set_ylabel('Sensor Value (from Amplitude column)')
    ax2.legend()
    ax2.grid(True, linestyle=':')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the process."""
    
    # --- IMPORTANT ---
    # Change this variable to point to your CSV file
    csv_filename = 'ddddfffa.csv' 
    # ---------------
    
    try:
        # 1. Read the data
        print(f"Reading data from {csv_filename}...")
        original_data = pd.read_csv(csv_filename)
        
        # 2. Rename columns to the standard names the script expects
        #    This makes it easier to work with, regardless of the
        #    original (and long) column names.
        original_data = original_data.rename(columns={
            'Index': 'timestamp',
            'Amplitude (0-4096)': 'value'
        })
        
        # Check if renaming worked (i.e., if 'value' column exists now)
        if 'value' not in original_data.columns or 'timestamp' not in original_data.columns:
            print("Error: Failed to find expected columns.")
            print("Please ensure your CSV has columns named 'Index' and 'Amplitude (0-4096)'.")
            return

        # 3. Clean the data
        print("Cleaning data...")
        cleaned_data = clean_sensor_data(original_data, max_val=4096)
        
        # 4. Save the cleaned data to a new file
        output_csv_filename = 'cleaned_data.csv'
        print(f"Saving cleaned data to {output_csv_filename}...")
        
        # Rename columns back to original for saving
        data_to_save = cleaned_data.rename(columns={
            'timestamp': 'Index',
            'value': 'Amplitude (0-4096)'
        })
        
        # Save to new CSV, without the pandas index
        data_to_save.to_csv(output_csv_filename, index=False)
        
        # 5. Plot the results
        print("Plotting results...")
        plot_comparison(original_data, cleaned_data, max_val=4096)
        
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"File not found: '{csv_filename}'")
        print("Please make sure your CSV file is in the same directory as the script,")
        print("or update the 'csv_filename' variable at the top of the main() function.")
    except KeyError:
        # This error is now less likely due to the check above, but good to keep
        print(f"\n--- ERROR ---")
        print("Could not find the expected columns in the CSV.")
        print("Please ensure your CSV file has columns named exactly:")
        print("  'Index'")
        print("  'Amplitude (0-4096)'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()