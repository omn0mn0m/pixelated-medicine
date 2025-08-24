import pandas as pd
import pingouin as pg

def calculate_icc(df, score_type):
    print(f"--- Calculating ICC for {score_type} Scores ---")

    try:
        rater1_col = f'Reviewer_1_{score_type}_Score'
        rater2_col = f'Reviewer_2_{score_type}_Score'
        scores_df = df[[rater1_col, rater2_col]].copy()
    except KeyError:
        # Can occur if independent reviewer reformats their spreadsheet at all
        print(f"Error: Columns for '{score_type}' not found. Please check column names.")
        print("Expected names:", rater1_col, "and", rater2_col)
        return

    # Prepare data for calculation
    scores_df['Subject'] = range(len(scores_df))
    long_df = pd.melt(scores_df, id_vars='Subject', var_name='Rater', value_name='Score')

    # Calculate ICC
    icc = pg.intraclass_corr(data=long_df, targets='Subject', raters='Rater', ratings='Score')

    # Interpret and print results (no Excel sheet because this is a quick validation)
    icc.set_index('Type', inplace=True)

    single_rater_icc = icc.loc['ICC1']

    print(f"ICC Report for {score_type} Scores:")
    print(single_rater_icc)
    print("\nInterpretation:")
    icc_val = single_rater_icc['ICC']
    if icc_val < 0.5:
        print(f"The ICC value of {icc_val:.3f} suggests POOR reliability.")
    elif icc_val < 0.75:
        print(f"The ICC value of {icc_val:.3f} suggests MODERATE reliability.")
    elif icc_val < 0.9:
        print(f"The ICC value of {icc_val:.3f} suggests GOOD reliability.")
    else:
        print(f"The ICC value of {icc_val:.3f} suggests EXCELLENT reliability.")
    print("-" * 40 + "\n")


# Main script
try:
    # Load file from ./data/
    file_path = './data/IRR_Master_Sheet_For_Analysis.xlsx'
    master_df = pd.read_excel(file_path)

    # Calculate ICC for Treatment Scores
    calculate_icc(master_df, 'Treatment')

    # Calculate ICC for Recovery Scores
    calculate_icc(master_df, 'Recovery')
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the CSV file is in the correct location.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

