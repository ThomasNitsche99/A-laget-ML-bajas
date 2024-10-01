import pandas as pd
import argparse

def convert_delimiter(input_file, output_file, old_delimiter='|', new_delimiter=','):
    """
    Converts a CSV file with a custom delimiter to a new CSV file with a comma delimiter.
    
    Parameters:
        input_file (str): Path to the input CSV file with a custom delimiter.
        output_file (str): Path to save the converted CSV file.
        old_delimiter (str): The current delimiter used in the input file. Default is '|'.
        new_delimiter (str): The desired delimiter for the output file. Default is ','.
    """
    try:
        # Read the input file with the specified delimiter
        df = pd.read_csv(input_file, delimiter=old_delimiter)
        
        # Save the dataframe to a new CSV file with the desired delimiter
        df.to_csv(output_file, index=False, sep=new_delimiter)
        
        print(f"File has been successfully converted and saved as: {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser to take input and output file paths
    parser = argparse.ArgumentParser(description="Convert CSV file delimiter.")
    parser.add_argument("input_file", help="Path to the input CSV file with custom delimiter.")
    parser.add_argument("output_file", help="Path to save the converted CSV file.")
    parser.add_argument("--old_delimiter", default='|', help="Current delimiter used in the input file (default: '|').")
    parser.add_argument("--new_delimiter", default=',', help="Desired delimiter for the output file (default: ',').")

    # Parse arguments
    args = parser.parse_args()
    
    # Call the function to convert delimiters
    convert_delimiter(args.input_file, args.output_file, args.old_delimiter, args.new_delimiter)
