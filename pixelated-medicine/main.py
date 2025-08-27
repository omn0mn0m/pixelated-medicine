# =======================================
# Imports
# =======================================
import os
import json
from io import StringIO
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from igdb.wrapper import IGDBWrapper
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import scipy as sp
import scikit_posthocs as posthoc

# =======================================
# Constants
# =======================================
AUTH_URL = 'https://id.twitch.tv/oauth2/token'
DATA_FILE_PATH = './data/Medical Encounter Data.xlsx'
TABLES_OUTPUT_PATH = "./data/Tables.xlsx"
SUPPLEMENTAL_OUTPUT_PATH = "./data/Supplemental_Results.xlsx"
MIN_CATEGORY_COUNT = 10
P_VALUE_THRESHOLD = 0.05

# Column Names
TREATMENT_ACCURACY = 'treatment_accuracy'
RECOVERY_ACCURACY = 'recovery_accuracy'
IS_FICTIONAL = 'is_fictional'

# Categories for Analysis
REGRESSION_CATEGORIES = [
    'age_rating_category', 'age_rating_descriptions', 'game_modes',
    'genres', 'platforms', 'player_perspectives', 'themes',
]

# Replacement dictionary for table outputs
HUMAN_READABLE_NAMES = {
    # Predictor Variables
    'age_rating_category': 'ESRB Rating',
    'age_rating_descriptions': 'ESRB Content Descriptors',
    'game_modes': 'Game Modes',
    'genres': 'Genres',
    'platforms': 'Platforms',
    'player_perspectives': 'Player Perspectives',
    'themes': 'Themes',

    # Outcome Variables
    'treatment_accuracy': 'Treatment Accuracy',
    'recovery_accuracy': 'Recovery Accuracy',
    'is_fictional': 'Is Fictional'
}

# =======================================
# Manuscript Table Formatting Functions
# =======================================
def format_p_values_for_manuscript(p_value_column):
    """Converts a p-value column to a string, representing values < 0.001 as '< 0.001'."""
    return p_value_column.apply(lambda p: '< 0.001' if p < 0.001 else f"{p:.3f}")

def format_regression_for_manuscript(model_results, title, full_results=False):
    """
    Formats regression results into a manuscript-friendly table.
    If full_results is True, it keeps all variables, including non-significant ones and intercepts.
    """
    results_df = pd.DataFrame({
        "Variable": model_results.params.index,
        "Coefficient": model_results.params.values,
        "Std. Error": model_results.bse.values,
        "P-Value": model_results.pvalues.values,
        "CI Lower": np.exp(model_results.conf_int()[0]),
        "CI Upper": np.exp(model_results.conf_int()[1]),
    })

    if isinstance(model_results.model, (sm.Logit, OrderedModel)):
        results_df["Odds Ratio"] = np.exp(model_results.params.values)

    # For manuscript tables, we clean up variable names and remove intercepts/cut-points
    if not full_results:
        results_df['Variable'] = results_df['Variable'].str.extract(r'\[T\.(.*?)\]')
        results_df.dropna(subset=['Variable'], inplace=True)
    else:
        # For full results, we just clean up the names that match the pattern
        results_df['Variable'] = results_df['Variable'].str.extract(r'\[T\.(.*?)\]').fillna(results_df['Variable'])

    numeric_cols = results_df.select_dtypes(include=np.number).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(3)

    if "Odds Ratio" in results_df.columns:
        results_df["Odds Ratio"] = results_df["Odds Ratio"].round(2)

    results_df.name = title
    return results_df

def format_icd10_table(icd10_counts):
    """Formats Table: Top 10 ICD-10 Categories."""
    top_10 = icd10_counts.head(10).reset_index()
    top_10.columns = ['ICD-10', 'Count']
    total_count = icd10_counts.sum()
    top_10['Percent'] = (top_10['Count'] / total_count * 100).round(2).astype(str) + '%'
    top_10.name = "Top 10 ICD-10"
    return top_10

def format_summary_tables(games_df, injuries_df):
    """Formats Tables: Summary of Game Metadata."""
    def summarize_df(df, name):
        summary_data = []
        for col in REGRESSION_CATEGORIES:
            if col in df.columns:
                exploded = df[col].explode()
                count = exploded.count()
                unique = exploded.nunique()
                top = exploded.mode()[0] if not exploded.mode().empty else 'N/A'
                freq = (exploded == top).sum()
                summary_data.append([col, count, unique, top, freq])
        summary_df = pd.DataFrame(summary_data, columns=['Game Characteristic', 'Count', 'Unique', 'Top', 'Frequency'])
        summary_df['Game Characteristic'] = summary_df['Game Characteristic'].map(HUMAN_READABLE_NAMES).fillna(summary_df['Game Characteristic'])
        summary_df.name = name
        return summary_df
    game_summary_table = summarize_df(games_df, "Game Summary")
    injury_summary_table = summarize_df(injuries_df, "Injury Summary")
    return game_summary_table, injury_summary_table

def format_accuracy_mean_tables(treatment_means, recovery_means):
    """Formats Tables: Average Accuracy by Genre and Theme."""
    def create_accuracy_table(means_dict, accuracy_col_name, name):
        genre_means = means_dict.get('genres', pd.DataFrame()).round(1)
        theme_means = means_dict.get('themes', pd.DataFrame()).round(1)

        genre_means['Category Type'] = 'Genre'
        theme_means['Category Type'] = 'Theme'

        genre_means.index.name = 'Category Name'
        theme_means.index.name = 'Category Name'

        genre_means.rename(columns={accuracy_col_name: 'Average Accuracy'}, inplace=True)
        theme_means.rename(columns={accuracy_col_name: 'Average Accuracy'}, inplace=True)

        combined = pd.concat([genre_means, theme_means]).reset_index()

        combined.sort_values(by=['Category Type', 'Category Name'], inplace=True)
        combined.loc[combined.duplicated(subset=['Category Type']), 'Category Type'] = ''

        combined = combined[['Category Type', 'Category Name', 'Average Accuracy']]
        combined.name = name
        return combined

    treatment_acc_table = create_accuracy_table(treatment_means, TREATMENT_ACCURACY, "Avg Treatment Accuracy")
    recovery_acc_table = create_accuracy_table(recovery_means, RECOVERY_ACCURACY, "Avg Recovery Accuracy")
    return treatment_acc_table, recovery_acc_table

def format_kruskal_wallis_table(results_list):
    """Formats Table: Kruskal-Wallis Test Results."""
    kruskal_data = [r for r in results_list if r['analysis_type'] == 'Kruskal-Wallis']
    if not kruskal_data:
        return pd.DataFrame()

    kruskal_df = pd.DataFrame(kruskal_data)
    kruskal_df['p_value'] = format_p_values_for_manuscript(kruskal_df['p_value'])
    kruskal_df['predictor'] = kruskal_df['predictor'].map(HUMAN_READABLE_NAMES)
    kruskal_df['outcome'] = kruskal_df['outcome'].map(HUMAN_READABLE_NAMES)

    kruskal_df.rename(columns={
        'outcome': 'Accuracy Type', 'predictor': 'Game Characteristic',
        'statistic': 'H-statistic', 'p_value': 'p-value', 'eta_squared': 'Eta Squared'
    }, inplace=True)

    final_df = kruskal_df[['Accuracy Type', 'Game Characteristic', 'H-statistic', 'p-value', 'Eta Squared']].copy()
    numeric_cols = final_df.select_dtypes(include=np.number).columns
    final_df[numeric_cols] = final_df[numeric_cols].round(3)

    final_df.sort_values(by=['Accuracy Type', 'Game Characteristic'], inplace=True)
    final_df.loc[final_df.duplicated(subset=['Accuracy Type']), 'Accuracy Type'] = ''

    final_df.name = "Kruskal-Wallis Results"
    return final_df

def format_dunn_tables(results_list):
    """Formats Dunn's Post-Hoc results into two separate tables, avoiding redundant pairs."""
    dunn_data = [r for r in results_list if r['analysis_type'] == 'Dunn']
    all_significant = []
    for result in dunn_data:
        dunn_df = result['results_df']
        mask = np.triu(np.ones_like(dunn_df, dtype=bool), k=1)
        upper_triangle_df = dunn_df.where(mask)
        melted = upper_triangle_df.reset_index().melt(id_vars='index', var_name='Comparison Group 2', value_name='p-value')
        melted.dropna(subset=['p-value'], inplace=True)
        melted.rename(columns={'index': 'Comparison Group 1'}, inplace=True)
        significant = melted[melted['p-value'] < P_VALUE_THRESHOLD].copy()
        if not significant.empty:
            significant['Analysis Type'] = HUMAN_READABLE_NAMES.get(result['outcome'])
            significant['Category'] = HUMAN_READABLE_NAMES.get(result['predictor'])
            all_significant.append(significant)

    if not all_significant:
        return pd.DataFrame(), pd.DataFrame()

    final_table = pd.concat(all_significant, ignore_index=True)
    final_table['p-value'] = format_p_values_for_manuscript(final_table['p-value'])

    treatment_hr = HUMAN_READABLE_NAMES.get(TREATMENT_ACCURACY)
    recovery_hr = HUMAN_READABLE_NAMES.get(RECOVERY_ACCURACY)

    treatment_table = final_table[final_table['Analysis Type'] == treatment_hr].copy()
    recovery_table = final_table[final_table['Analysis Type'] == recovery_hr].copy()

    if not treatment_table.empty:
        treatment_table.sort_values(by=['Category', 'Comparison Group 1', 'Comparison Group 2'], inplace=True)
        treatment_table.loc[treatment_table.duplicated(subset=['Category']), 'Category'] = ''
        treatment_table = treatment_table[['Category', 'Comparison Group 1', 'Comparison Group 2', 'p-value']]
    treatment_table.name = "Dunn Post-Hoc (Treatment)"

    if not recovery_table.empty:
        recovery_table.sort_values(by=['Category', 'Comparison Group 1', 'Comparison Group 2'], inplace=True)
        recovery_table.loc[recovery_table.duplicated(subset=['Category']), 'Category'] = ''
        recovery_table = recovery_table[['Category', 'Comparison Group 1', 'Comparison Group 2', 'p-value']]
    recovery_table.name = "Dunn Post-Hoc (Recovery)"

    return treatment_table, recovery_table

def format_olr_tables(results_list):
    """Formats significant Ordinal Logistic Regression results into two tables."""
    olr_results = [r for r in results_list if r['analysis_type'] == 'OLR']
    significant_dfs = []
    for res in olr_results:
        formatted = format_regression_for_manuscript(res['model_results'], "")
        significant = formatted[formatted['P-Value'] < P_VALUE_THRESHOLD].copy()

        if not significant.empty:
            significant['P-Value'] = format_p_values_for_manuscript(significant['P-Value'])
            significant['Predictor'] = HUMAN_READABLE_NAMES.get(res['predictor'])
            significant['Accuracy'] = res['outcome']
            significant_dfs.append(significant)

    if not significant_dfs:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(significant_dfs)
    combined.rename(columns={'Variable': 'Category'}, inplace=True)
    combined['Odds Ratio (95% CI)'] = combined.apply(
        lambda row: f"{row['Odds Ratio']:.2f} [{row['CI Lower']:.2f}, {row['CI Upper']:.2f}]", axis=1)
    combined.rename(columns={'P-Value': 'p-value'}, inplace=True)
    final_cols = ['Predictor', 'Category', 'Odds Ratio (95% CI)', 'p-value']

    treatment_table = combined[combined['Accuracy'] == TREATMENT_ACCURACY][final_cols].copy()
    recovery_table = combined[combined['Accuracy'] == RECOVERY_ACCURACY][final_cols].copy()

    if not treatment_table.empty:
        treatment_table.sort_values(by=['Predictor', 'Category'], inplace=True)
        treatment_table.loc[treatment_table.duplicated(subset=['Predictor']), 'Predictor'] = ''

    if not recovery_table.empty:
        recovery_table.sort_values(by=['Predictor', 'Category'], inplace=True)
        recovery_table.loc[recovery_table.duplicated(subset=['Predictor']), 'Predictor'] = ''

    treatment_table.name = "OLR (Treatment Accuracy)"
    recovery_table.name = "OLR (Recovery Accuracy)"
    return treatment_table, recovery_table

def format_blr_table(results_list):
    """Formats significant Binary Logistic Regression results into a manuscript table."""
    blr_results = [r for r in results_list if r['analysis_type'] == 'BLR']
    significant_dfs = []
    for res in blr_results:
        formatted = format_regression_for_manuscript(res['model_results'], "")
        significant = formatted[formatted['P-Value'] < P_VALUE_THRESHOLD].copy()

        if not significant.empty:
            significant['P-Value'] = format_p_values_for_manuscript(significant['P-Value'])
            significant['Predictor'] = HUMAN_READABLE_NAMES.get(res['predictor'])
            significant_dfs.append(significant)

    if not significant_dfs:
        blr_table = pd.DataFrame()
        blr_table.name = "BLR (Is Fictional)"
        return blr_table

    combined = pd.concat(significant_dfs)
    combined.sort_values(by=['Predictor', 'Variable'], inplace=True)
    combined.loc[combined.duplicated(subset=['Predictor']), 'Predictor'] = ''
    combined['Odds Ratio (95% CI)'] = combined.apply(
        lambda row: f"{row['Odds Ratio']:.2f} [{row['CI Lower']:.2f}, {row['CI Upper']:.2f}]", axis=1)
    combined.rename(columns={'Variable': 'Category', 'P-Value': 'p-value'}, inplace=True)
    final_cols = ['Predictor', 'Category', 'Odds Ratio (95% CI)', 'p-value']
    blr_table = combined[final_cols]
    blr_table.name = "BLR (Is Fictional)"
    return blr_table

# =======================================
# Helper Functions
# =======================================
def fetch_igdb_game_data(wrapper, game_ids, query_limit=10):
    """Fetches detailed game information from the IGDB API."""
    game_info, fields = [], """
        name,age_ratings.organization.name,age_ratings.rating_category.rating,
        age_ratings.rating_content_descriptions.description,aggregated_rating,category,
        franchise.name,game_engines.name,game_modes.name,genres.name,
        involved_companies.company.name,keywords.name,multiplayer_modes,first_release_date,
        platforms.name,player_perspectives.name,storyline,summary,themes.name,total_rating
    """
    for i in range(0, len(game_ids), query_limit):
        batch = game_ids[i:i + query_limit]
        game_query_str = ','.join(map(str, batch.astype(int)))
        search_string = f"fields {fields}; where id=({game_query_str}); limit {query_limit};"
        try:
            results = json.loads(wrapper.api_request('games', search_string))
            if results: game_info.extend(results)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"An error occurred during API request: {e}")
    games = pd.DataFrame(game_info)
    for col in ['franchise', 'game_engines', 'genres', 'themes', 'game_modes', 'keywords', 'platforms', 'player_perspectives']:
        games[col] = games[col].apply(lambda d: [i.get('name') for i in d] if isinstance(d, list) else (d.get('name') if isinstance(d, dict) else d))
    games.involved_companies = games.involved_companies.apply(lambda d: [c.get('company', {}).get('name') for c in d if c.get('company')] if isinstance(d, list) else None)
    games.age_ratings = games.age_ratings.apply(lambda ratings: next(({'organization': 'ESRB', 'category': r.get('rating_category', {}).get('rating'), 'descriptions': [d.get('description') for d in r.get('rating_content_descriptions', []) if d.get('description')]} for r in ratings if isinstance(r, dict) and r.get('organization', {}).get('name') == 'ESRB'), None) if isinstance(ratings, list) else None)
    age_ratings_df = pd.json_normalize(games['age_ratings']).add_prefix('age_rating_')
    return games.join(age_ratings_df).drop(columns=['age_ratings'])

def get_descriptive_stats(dataframe):
    """Calculates descriptive statistics, exploding list columns."""
    list_cols = [c for c in dataframe.columns if dataframe[c].apply(isinstance, args=(list,)).any()]
    direct_stats = dataframe.drop(columns=list_cols).describe(include='all')
    exploded_stats = pd.DataFrame({c: dataframe[c].explode().describe(include='all') for c in list_cols})
    return pd.concat([direct_stats, exploded_stats], axis=1)

def handle_sparsity(df, category_col, min_count):
    """Combines sparse categories into an 'Other' group."""
    counts = df[category_col].value_counts()
    to_combine = counts[counts < min_count].index
    if len(to_combine) > 1:
        print(f"  Combining {len(to_combine)} sparse groups in '{category_col}' into 'Other'.")
        df[category_col] = df[category_col].replace(to_combine, 'Other')
    return df, to_combine.tolist()

def handle_separation(df, x_header, y_header):
    """Removes categories causing complete separation."""
    crosstab = pd.crosstab(df[x_header], df[y_header])
    problem_cats = crosstab[(crosstab > 0).sum(axis=1) == 1].index
    if not problem_cats.empty:
        print(f"  Excluding {len(problem_cats)} categories in '{x_header}' causing separation.")
        df = df[~df[x_header].isin(problem_cats)]
    return df, problem_cats.tolist()

# =======================================
# Core Analysis Function
# =======================================
def run_statistical_analyses(categories, merged_df, merged_recovery_df, merged_logistic_df):
    """Performs all statistical analyses and returns results in a structured list."""
    all_results = []
    print("Running Binary Logistic Regressions...")
    for category in categories:
        df_blr = merged_logistic_df[[category, IS_FICTIONAL]].copy()
        if df_blr[category].apply(isinstance, args=(list,)).any(): df_blr = df_blr.explode(category)

        # Data cleanup
        df_blr.dropna(subset=[category, IS_FICTIONAL], inplace=True)
        if df_blr.empty or df_blr[category].nunique() < 2 or df_blr[IS_FICTIONAL].nunique() < 2: continue
        df_blr, combined = handle_sparsity(df_blr, category, MIN_CATEGORY_COUNT)
        df_blr, separated = handle_separation(df_blr, category, IS_FICTIONAL)
        if df_blr.empty or df_blr[category].nunique() < 2: continue

        ref_category = df_blr[category].mode()[0]

        try:
            formula = f"is_fictional ~ C({category}, Treatment(reference='{ref_category}'))"
            model = smf.logit(formula, data=df_blr).fit(disp=0)
            all_results.append({'analysis_type': 'BLR', 'predictor': category, 'outcome': IS_FICTIONAL, 'model_results': model, 'reference_category': ref_category, 'combined_groups': combined, 'separated_groups': separated})
        except Exception as e:
            print(f"  BLR failed for '{category}'. Error: {e}")

    print("\nRunning Ordinal Regressions and Kruskal-Wallis tests...")
    for outcome in [TREATMENT_ACCURACY, RECOVERY_ACCURACY]:
        base_df = merged_recovery_df if outcome == RECOVERY_ACCURACY else merged_df

        for category in categories:
            df_analysis = base_df[[category, outcome]].copy()
            if df_analysis[category].apply(isinstance, args=(list,)).any():
                df_analysis = df_analysis.explode(category)

            df_analysis[outcome] = pd.to_numeric(df_analysis[outcome], errors='coerce')
            df_analysis.dropna(subset=[outcome, category], inplace=True)

            if df_analysis.empty or df_analysis[category].nunique() < 2: continue

            df_analysis, combined = handle_sparsity(df_analysis, category, MIN_CATEGORY_COUNT)
            df_analysis, separated = handle_separation(df_analysis, category, outcome)

            if df_analysis.empty or df_analysis[category].nunique() < 2: continue

            try:
                ref_cat = df_analysis[category].value_counts().index[0]
                formula_olr = f"{outcome} ~ C({category}, Treatment(reference='{ref_cat}'))"
                model_olr = OrderedModel.from_formula(formula_olr, df_analysis, distr='logit').fit(method='bfgs', disp=0)
                all_results.append({
                    'analysis_type': 'OLR',
                    'predictor': category,
                    'outcome': outcome,
                    'model_results': model_olr,
                    'reference_category': ref_cat,
                    'combined_groups': combined,
                    'separated_groups': separated
                })
            except Exception as e:
                print(f"    OLR failed for {category} vs {outcome}. Error: {e}")

            groups = df_analysis.groupby(category)[outcome].apply(list)
            stat, p_val = sp.stats.kruskal(*groups)
            k, n = len(groups), len(df_analysis)
            eta_sq = (stat - k + 1) / (n - k) if n > k else 0
            all_results.append({'analysis_type': 'Kruskal-Wallis', 'predictor': category, 'outcome': outcome, 'statistic': stat, 'p_value': p_val, 'eta_squared': eta_sq})

            if p_val < P_VALUE_THRESHOLD:
                dunn_df = posthoc.posthoc_dunn(df_analysis, val_col=outcome, group_col=category, p_adjust='holm')
                all_results.append({'analysis_type': 'Dunn', 'predictor': category, 'outcome': outcome, 'results_df': dunn_df})
    return all_results

# =======================================
# Main Execution Block
# =======================================
if __name__ == "__main__":
    # --- Setup and Data Loading ---
    load_dotenv()
    CLIENT_ID, CLIENT_SECRET = os.getenv('IGDB_CLIENT'), os.getenv('IGDB_TOKEN')
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("IGDB environment variables must be set.")

    response = requests.post(AUTH_URL, params={'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET, 'grant_type': 'client_credentials'})
    response.raise_for_status()
    wrapper = IGDBWrapper(CLIENT_ID, response.json()['access_token'])

    injuries = pd.read_excel(DATA_FILE_PATH)
    injuries[["Treatment Accuracy", "Recovery Accuracy"]] = injuries[["Treatment Accuracy", "Recovery Accuracy"]].apply(lambda x: pd.to_numeric(x.astype(str).str[0]))
    injuries_for_blr = injuries.rename(columns=lambda x: x.lower().replace(' ', '_')).assign(is_fictional=lambda df: (df.treatment_accuracy == 0).astype(int))

    injuries = injuries[injuries['Treatment Accuracy'] != 0]
    injuries = injuries.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_')).assign(icd_10=lambda df: df['icd_10'].str.split('.').str[0].str.split(','))

    games = fetch_igdb_game_data(wrapper, injuries['igdb_id'].unique())

    merged_logistic_base = injuries_for_blr.merge(games, left_on='igdb_id', right_on='id')
    merged_base = injuries[injuries['treatment_accuracy'] != 0].merge(games, left_on='igdb_id', right_on='id')
    merged_treatment = merged_base.copy()
    merged_recovery = merged_base[merged_base['recovery_accuracy'] != 0].copy()

    os.makedirs(os.path.dirname(TABLES_OUTPUT_PATH), exist_ok=True)

    # --- Descriptive Statistics ---
    icd10_counts = injuries["icd_10"].explode().value_counts()
    all_treatment_means, all_recovery_means = {}, {}
    for category in ['age_rating_descriptions', 'genres', 'themes']:
        all_treatment_means[category] = merged_treatment.explode(category).groupby(category)[[TREATMENT_ACCURACY]].mean()
        all_recovery_means[category] = merged_recovery.explode(category).groupby(category)[[RECOVERY_ACCURACY]].mean()

    # --- Spearman Correlation ---
    corr_data = merged_recovery[[TREATMENT_ACCURACY, RECOVERY_ACCURACY, 'total_rating']].dropna().astype(int)
    spearman_matrix, p_value_matrix = sp.stats.spearmanr(corr_data)
    spearman_df = pd.DataFrame(spearman_matrix, index=corr_data.columns, columns=corr_data.columns)
    pval_df = pd.DataFrame(p_value_matrix, index=corr_data.columns, columns=corr_data.columns)

    # --- Run Core Statistical Analyses ---
    all_analysis_results = run_statistical_analyses(REGRESSION_CATEGORIES, merged_treatment, merged_recovery, merged_logistic_base)

    # --- Generate Manuscript Tables ---
    print("\nGenerating manuscript tables...")
    icd10_table = format_icd10_table(icd10_counts)
    game_summary_table, injury_summary_table = format_summary_tables(games, merged_base)
    treatment_acc_table, recovery_acc_table = format_accuracy_mean_tables(all_treatment_means, all_recovery_means)

    kruskal_wallis_table = format_kruskal_wallis_table(all_analysis_results)
    dunn_treatment_table, dunn_recovery_table = format_dunn_tables(all_analysis_results)

    olr_treatment_table, olr_recovery_table = format_olr_tables(all_analysis_results)
    blr_table = format_blr_table(all_analysis_results)

    manuscript_tables = [
        icd10_table, game_summary_table, injury_summary_table,
        treatment_acc_table, recovery_acc_table, kruskal_wallis_table,
        dunn_treatment_table, dunn_recovery_table,
        olr_treatment_table, olr_recovery_table,
        blr_table
    ]

    # --- Write Manuscript Tables to Tables.xlsx ---
    print(f"Writing manuscript tables to {TABLES_OUTPUT_PATH}...")
    with pd.ExcelWriter(TABLES_OUTPUT_PATH) as writer:
        for table in manuscript_tables:
            if table is not None and not table.empty:
                table.to_excel(writer, sheet_name=table.name, index=False)

    # --- Write Full and Supplementary Results to Supplemental_Results.xlsx ---
    print(f"Writing all supplementary data and full results to {SUPPLEMENTAL_OUTPUT_PATH}...")
    with pd.ExcelWriter(SUPPLEMENTAL_OUTPUT_PATH) as writer:
        # Write supplementary data
        merged_base.to_excel(writer, sheet_name='Supp - Merged Data', index=False)
        games.to_excel(writer, sheet_name='Supp - Games Data', index=False)

        get_descriptive_stats(merged_base).to_excel(writer, sheet_name='Supp - Descriptive Stats')
        pd.concat(all_treatment_means).to_excel(writer, sheet_name="Supp - All Treatment Means")
        pd.concat(all_recovery_means).to_excel(writer, sheet_name="Supp - All Recovery Means")

        spearman_df.to_excel(writer, sheet_name='Supp - Spearman R')
        pval_df.to_excel(writer, sheet_name='Supp - Spearman R', startrow=len(spearman_df)+2)
        pd.DataFrame({'N': [len(corr_data)]}).to_excel(writer, sheet_name='Supp - Spearman R', startrow=len(spearman_df)+len(pval_df)+4, index=False)

        # Write full, unabridged analysis results
        for res_type in ['BLR', 'OLR', 'Dunn']:
            results = [r for r in all_analysis_results if r['analysis_type'] == res_type]
            for res in results:
                predictor_hr = HUMAN_READABLE_NAMES.get(res['predictor'], res['predictor'])
                outcome_hr = HUMAN_READABLE_NAMES.get(res['outcome'], res['outcome'])
                sheet_name = f"Full - {res_type} {predictor_hr}-{outcome_hr}"[:31]

                if res_type in ['BLR', 'OLR']:
                    # Use full_results=True to get unabridged output
                    full_regression_output = format_regression_for_manuscript(res['model_results'], "", full_results=True)
                    full_regression_output.to_excel(writer, sheet_name=sheet_name, index=False)
                elif res_type == 'Dunn':
                    dunn_df = res['results_df']
                    melted_dunn = dunn_df.reset_index().melt(id_vars='index', var_name='Group 2', value_name='p-value')
                    melted_dunn.rename(columns={'index': 'Group 1'}, inplace=True)
                    melted_dunn.dropna(subset=['p-value'], inplace=True)
                    melted_dunn.sort_values(by=['Group 1', 'Group 2'], inplace=True)
                    melted_dunn.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Analysis complete. All files have been saved.")
