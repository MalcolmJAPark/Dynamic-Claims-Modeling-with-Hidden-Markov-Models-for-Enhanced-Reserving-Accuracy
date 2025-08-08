import pandas as pd

def drop_null_entries(input_path: str, output_path: str):
    """
    Loads the combined data CSV, drops any rows with nulls in the specified necessary columns,
    and writes the cleaned data to a new CSV.
    """
    # 1. Load data, parsing date columns
    df = pd.read_csv(input_path, parse_dates=['dateOfLoss', 'asOfDate'])
    
    # 2. Specify the columns required for the HMM model
    necessary_cols = [
        'id',
        'dateOfLoss',
        'asOfDate',
        'policyCount',
        'netBuildingPaymentAmount',
        'netContentsPaymentAmount',
        'state',
        'ratedFloodZone',
        'elevationDifference',
        'postFIRMConstructionIndicator',
        'primaryResidenceIndicator'
    ]
    
    # 3. Drop rows with nulls in any of these columns
    before_count = len(df)
    df_clean = df.dropna(subset=necessary_cols)
    after_count = len(df_clean)
    
    # 4. Save the cleaned DataFrame
    df_clean.to_csv(output_path, index=False)
    
    print(f"Dropped {before_count - after_count} entries. Cleaned data saved to '{output_path}'.")

if __name__ == "__main__":
    drop_null_entries("combined_data.csv", "combined_data_clean.csv")


## example usage python3 drop_null_entries.py combined_data.csv combined_data_clean.csv
