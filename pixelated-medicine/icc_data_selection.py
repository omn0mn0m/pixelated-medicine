import pandas as pd
import numpy as np

# Main data file
INPUT_FILE_PATH = './data/Medical Encounter Data.xlsx'
# The name for the new file that will be sent to the second reviewer (blinded)
OUTPUT_FILE_BLINDED = './data/IRR_Blinded_For_Reviewer2.xlsx'
# The name for the new file to compile scores for analysis
OUTPUT_FILE_MASTER = './data/IRR_Master_Sheet_For_Analysis.xlsx'
# The percentage of the total data to use for the validation sample
SAMPLE_FRACTION = 0.25
# A random seed to get the same "random" sample every time script is run
RANDOM_SEED = 42

def create_validation_sample(input_file, output_blinded, output_master, fraction, seed):
    """
    Loads medical encounter data, selects a random sample, and prepares two files:
    1. A blinded file for a second reviewer.
    2. A master file for the researcher to compile scores for IRR analysis.

    Args:
        input_file (str): Path to the source Excel file.
        output_blinded (str): Path to save the blinded Excel file for Reviewer 2.
        output_master (str): Path to save the master Excel file for the researcher.
        fraction (float): The fraction of the dataset to sample (e.g., 0.25 for 25%).
        seed (int): A random seed for reproducibility.
    """
    print(f"Loading original data from: {input_file}")
    # Load original data
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{input_file}'. Please check the path.")
        return

    # Ensure the dataframe is not empty
    if df.empty:
        print("ERROR: The source file is empty.")
        return

    # Calculate the number of samples to pull
    num_samples = int(np.ceil(len(df) * fraction))
    print(f"Original dataset has {len(df)} scenarios.")
    print(f"Creating a random sample of {num_samples} scenarios ({fraction:.0%})...")

    # Create the random sample using the reproducible random_state
    validation_sample_df = df.sample(n=num_samples, random_state=seed)

    # Creat blinded file
    print("Preparing the blinded sample file for the second reviewer...")

    columns_for_reviewer = [
        'Game', 'Timestamp', 'Character', 'Age', 'Setting', 'Chief Complaint',
        'Subjective', 'Objective', 'Assessment/ Plan', 'Recovery/ Follow-Up/ Effect of Treatment'
    ]

    final_columns_reviewer = [col for col in columns_for_reviewer if col in validation_sample_df.columns]

    blinded_df = validation_sample_df[final_columns_reviewer].copy()
    blinded_df['Reviewer_2_Treatment_Score'] = ''
    blinded_df['Reviewer_2_Recovery_Score'] = ''

    # Create master analysis file
    print("Preparing the master file for analysis...")

    columns_for_master = [
        'Game', 'Timestamp', 'Character', 'Age', 'Setting', 'Chief Complaint',
        'Subjective', 'Objective', 'Assessment/ Plan', 'Recovery/ Follow-Up/ Effect of Treatment',
        'Treatment Accuracy', 'Recovery Accuracy'
    ]

    final_columns_master = [col for col in columns_for_master if col in validation_sample_df.columns]

    master_df = validation_sample_df[final_columns_master].copy()

    # Clean the data entry for the scores (was using drop down with descriptions)
    for col in ['Treatment Accuracy', 'Recovery Accuracy']:
        if col in master_df.columns:
            # Convert to string, extract digits, then convert to a nullable integer
            master_df[col] = pd.to_numeric(
                master_df[col].astype(str).str.extract(r'(\d+)', expand=False),
                errors='coerce'
            ).astype('Int64')

    master_df['Reviewer_1_Treatment_Score'] = ''
    master_df['Reviewer_1_Recovery_Score'] = ''
    master_df['Reviewer_2_Treatment_Score'] = ''
    master_df['Reviewer_2_Recovery_Score'] = ''


    # Output files
    try:
        blinded_df.to_excel(output_blinded, index=False)
        master_df.to_excel(output_master, index=False)
        print(f"\nSUCCESS! Two files have been created:")
        print(f" -> BLINDED file for Reviewer 2: {output_blinded}")
        print(f" -> MASTER file for your analysis: {output_master}")
    except Exception as e:
        print(f"\nERROR: Could not save the files. Reason: {e}")

if __name__ == "__main__":
    # Create a dummy data directory and file for demonstration if they don't exist
    import os
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if not os.path.exists(INPUT_FILE_PATH):
        print("Creating a dummy input file for demonstration purposes...")
        dummy_data = pd.DataFrame({
            'Game': [f'Game {i//10 + 1}' for i in range(320)],
            'Timestamp': [f'00:{i%60:02d}:{i%60:02d}' for i in range(320)],
            'Character': [f'Character_{i+1}' for i in range(320)],
            'Age': [np.random.randint(18, 65) for _ in range(320)],
            'Setting': ['Field' if i % 2 == 0 else 'Hospital' for i in range(320)],
            'Chief Complaint': [f'Complaint {i+1}' for i in range(320)],
            'Subjective': ['Patient reports pain.' for _ in range(320)],
            'Objective': ['Visible wound.' for _ in range(320)],
            'Assessment/ Plan': ['Injury noted, plan to treat.' for _ in range(320)],
            'Recovery/ Follow-Up/ Effect of Treatment': ['Patient recovered fully.' for _ in range(320)],
            'Treatment Accuracy': [f'{x} - Description' for x in np.random.randint(0, 6, 320)],
            'Recovery Accuracy': [f'{x} - Description' for x in np.random.randint(0, 4, 320)]
        })
        dummy_data.to_excel(INPUT_FILE_PATH, index=False)

    # Create the files
    create_validation_sample(
        input_file=INPUT_FILE_PATH,
        output_blinded=OUTPUT_FILE_BLINDED,
        output_master=OUTPUT_FILE_MASTER,
        fraction=SAMPLE_FRACTION,
        seed=RANDOM_SEED
    )
