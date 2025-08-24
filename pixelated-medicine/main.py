import os
import json
from enum import Enum
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

# Load environment variables from .env file
load_dotenv()

# Dictionaries to convert string ratings to numerical values
treatment_accuracy_conversion = {
    "0 - Completely Fictional/ Unable to Assess": 0,
    "1 - Completely Inaccurate": 1,
    "2 - Mostly Inaccurate": 2,
    "3 - Half Accurate": 3,
    "4 - Mostly Accurate": 4,
    "5 - Completely Accurate": 5,
}

recovery_accuracy_conversion = {
    "0 - Unable to assess recovery time": 0,
    "1 - Recovery time too short": 1,
    "2 - Recovery time accurate": 2,
    "3 - Recovery time too long": 3,
}

access_token = requests.post('https://id.twitch.tv/oauth2/token?client_id={}&client_secret={}&grant_type=client_credentials'.format(os.getenv('IGDB_CLIENT', 'oops'), os.getenv('IGDB_TOKEN', 'oops'))).json()['access_token']

wrapper = IGDBWrapper(os.getenv('IGDB_CLIENT', 'oops'), access_token)

# Load and scrub data
injuries = pd.read_excel('./data/Medical Encounter Data.xlsx')
pd.set_option('future.no_silent_downcasting', True)
injuries = injuries.replace({"Treatment Accuracy": treatment_accuracy_conversion,
                  "Recovery Accuracy": recovery_accuracy_conversion,
                  })

# Save copy of injuries to include fictional
injuries_for_blr = injuries.copy()

# omit fictional or not assessible tx
injuries = injuries[injuries['Treatment Accuracy'] != 0]
injuries.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_'), inplace=True)
injuries['icd_10'] = injuries['icd_10'].str.split('.').str[0]
injuries['icd_10'] = injuries['icd_10'].str.split(',')
print(injuries)

icd10_counts = injuries["icd_10"].explode().value_counts()
print(icd10_counts)

# Get game data
unique_games = injuries['igdb_id'].unique()

game_info = []

for game in unique_games:
    search_string = ('fields name,age_ratings.organization.name,age_ratings.rating_category.rating,'
                     'age_ratings.rating_content_descriptions.description,aggregated_rating,category,'
                     'franchise.name,game_engines.name,game_modes.name,genres.name,'
                     'involved_companies.company.name,keywords.name,multiplayer_modes,first_release_date,'
                     'platforms.name,player_perspectives.name,storyline,summary,themes.name,total_rating; '
                     'where id={}; '
                     'limit 1;').format(game)

    results = json.loads(wrapper.api_request('games', search_string))

    if (results):
        game_info.append(results[0])
    else:
        print("Unable to find: {}".format(game))

games = pd.DataFrame(game_info)

def get_name(game_modes_dict):
    if isinstance(game_modes_dict, list):
        return [game_mode['name'] for game_mode in game_modes_dict]
    elif isinstance(game_modes_dict, dict):
        return game_modes_dict['name']
    else:
        return game_modes_dict

def get_company_name(company_list):
    return [company_dict['company']['name'] for company_dict in company_list]

def get_age_ratings(ratings):
    if not isinstance(ratings, list):
        return

    age_rating = {
        'organization': 'ESRB',
        'category': '',
        'descriptions': [],
    }

    for rating in ratings:
        if rating.get('organization').get('name') == 'ESRB':
            age_rating['category'] = rating.get('rating_category').get('rating')

            if 'rating_content_descriptions' in rating:
                descriptions = []
                for desc in rating['rating_content_descriptions']:
                    description_text = desc.get('description')

                    if description_text:
                        descriptions.append(description_text)
                age_rating['descriptions'] = descriptions

            break

    return age_rating

games.franchise = games.franchise.apply(get_name)
games.game_engines = games.game_engines.apply(get_name)
games.genres = games.genres.apply(get_name)
games.themes = games.themes.apply(get_name)
games.game_modes = games.game_modes.apply(get_name)
games.keywords = games.keywords.apply(get_name)
games.platforms = games.platforms.apply(get_name)
games.player_perspectives = games.player_perspectives.apply(get_name)
games.involved_companies = games.involved_companies.apply(get_company_name)
games.age_ratings = games.age_ratings.apply(get_age_ratings)

# Expand out age ratings
age_ratings_df = pd.json_normalize(games['age_ratings']).add_prefix('age_rating_')
games = games.join(age_ratings_df).drop(columns=['age_ratings'])

print(games)

# Basic injuries descriptive stats
injuries_descriptive = injuries.describe(include='all')
print(injuries_descriptive)

# Write out
with pd.ExcelWriter("./stats/Injuries.xlsx") as writer:
    injuries.to_excel(writer, sheet_name='Cleaned')
    injuries_descriptive.to_excel(writer, sheet_name='Descriptive')
    icd10_counts.to_excel(writer, sheet_name='ICD-10')

# Games descriptive stats
games_descriptive_base = games.describe(include='all')
print(games_descriptive_base)

categories_to_analyze = [
    'age_rating_category', 'age_rating_descriptions', 'franchise',
    'game_modes', 'genres', 'involved_companies', 'keywords',
    'platforms', 'player_perspectives', 'themes', 'game_engines'
]

exploded_stats_dict = {}

for category in categories_to_analyze:
    exploded_stats_dict[category] = games[category].explode().describe(include='all')

additional_descriptives = pd.DataFrame(exploded_stats_dict)
columns_to_drop = [col for col in categories_to_analyze if col in games_descriptive_base.columns]
games_descriptive_base = games_descriptive_base.drop(columns=columns_to_drop)
final_games_descriptive = pd.concat([games_descriptive_base, additional_descriptives], axis=1)

with pd.ExcelWriter("./stats/Games.xlsx") as writer:
    games.to_excel(writer, sheet_name='Cleaned')
    final_games_descriptive.to_excel(writer, sheet_name='Descriptive')

# Merge game data with injury data
merged = injuries.merge(games, left_on='igdb_id', right_on='id')
merged_recovery = injuries[injuries['recovery_accuracy'] != 0].merge(games, left_on='igdb_id', right_on='id') # specific dataframe for analyzing recovery since some encounters have no recovery
print(merged)
print(merged_recovery)

merged_descriptive = merged.describe(include='all')
print(merged_descriptive)

# Write out
with pd.ExcelWriter("./stats/Merged.xlsx") as writer:
    merged.to_excel(writer, sheet_name='Merged')
    merged_recovery.to_excel(writer, sheet_name='Merged Recovery')
    merged_descriptive.to_excel(writer, sheet_name='Descriptive')

descriptive_exploded = {}
for category in ['age_rating_category', 'age_rating_descriptions', 'franchise', 'game', 'game_modes', 'genres', 'involved_companies', 'keywords',
                 'platforms', 'player_perspectives', 'themes', 'game_engines']:
    descriptive_exploded[category] = merged[category].explode(category).describe(include='all')

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
    pd.concat(descriptive_exploded).to_excel(writer, sheet_name=f"Descriptive_{category}")

# Means for categories that need exploding
all_treatment_means = {}
all_recovery_means = {}

for category in ['age_rating_descriptions', 'genres', 'themes']:
    exploded_merged = merged.explode(category)
    exploded_merged_recovery = merged_recovery.explode(category)

    all_treatment_means[category] = exploded_merged.groupby(category)[['treatment_accuracy']].mean()
    all_recovery_means[category] = exploded_merged_recovery.groupby(category)[['recovery_accuracy']].mean()

# Write out means
all_treatment_means_df = pd.concat(all_treatment_means)
all_recovery_means_df = pd.concat(all_recovery_means)

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
    all_treatment_means_df.to_excel(writer, sheet_name="Consolidated Treatment Means")
    all_recovery_means_df.to_excel(writer, sheet_name="Consolidated Recovery Means")

# Spearman's correlation'
data_to_corr = pd.DataFrame({
    'treatment_accuracy': merged_recovery['treatment_accuracy'].astype(int),
    'recovery_accuracy': merged_recovery['recovery_accuracy'].astype(int),
    'total_rating': merged_recovery['total_rating']
}).dropna()

spearman_matrix, p_value_matrix = sp.stats.spearmanr(data_to_corr)
sample_size_N = len(data_to_corr)

spearman_matrix_df = pd.DataFrame(spearman_matrix, index=data_to_corr.columns, columns=data_to_corr.columns)
p_value_matrix_df = pd.DataFrame(p_value_matrix, index=data_to_corr.columns, columns=data_to_corr.columns)

sheet_name = 'Spearman_R_Analysis'

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
    current_row = 0

    spearman_matrix_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
    current_row += len(spearman_matrix_df) + 3

    p_value_matrix_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
    current_row += len(p_value_matrix_df)

    pd.DataFrame({'Sample Size (N)': [sample_size_N]}).to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=current_row,
        index=False
    )

# Binary logistic regression
injuries_for_blr.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_'), inplace=True)
injuries_for_blr['is_fictional'] = (injuries_for_blr['treatment_accuracy'] == 0).astype(int)
merged_logistic_base = injuries_for_blr.merge(games, left_on='igdb_id', right_on='id')

blr_categories = [
    'age_rating_category', 'age_rating_descriptions', 'game_modes',
    'genres', 'platforms', 'player_perspectives', 'themes', 'keywords'
]
MIN_CATEGORY_COUNT_BLR = 10

blr_other_details = {}
blr_separation_details = {}
blr_references = {}

for category in blr_categories:
    df_blr = merged_logistic_base[[category, 'is_fictional']].copy()

    if df_blr[category].apply(isinstance, args=(list,)).any():
        df_blr = df_blr.explode(category)

    df_blr.dropna(subset=[category, 'is_fictional'], inplace=True)

    # Skip if not enough data
    if df_blr.empty or df_blr[category].nunique() < 2 or df_blr['is_fictional'].nunique() < 2:
        print(f"  Skipping {category} due to insufficient data for modeling.")
        continue

    # Handle sparse categories
    value_counts = df_blr[category].value_counts()
    to_combine = value_counts[value_counts < MIN_CATEGORY_COUNT_BLR].index
    if not to_combine.empty:
        blr_other_details[category] = to_combine.tolist()

    if len(to_combine) > 1:
        print(f"  Combining {len(to_combine)} sparse groups in '{category}' into 'Other'.")
        df_blr[category] = df_blr[category].replace(to_combine, 'Other')

    # After combining, re-check group count
    if df_blr[category].nunique() < 2:
        print(f"  Skipping {category} after combining sparse groups (only one group left).")
        continue

    crosstab_df = pd.crosstab(df_blr[category], df_blr['is_fictional'])
    problem_categories = crosstab_df[(crosstab_df > 0).sum(axis=1) == 1]

    if not problem_categories.empty:
        problem_category_names = problem_categories.index
        blr_separation_details[category] = problem_category_names.tolist()
        print(f"  Found {len(problem_category_names)} categories causing complete separation: {list(problem_category_names)}")
        print("  Excluding them from this specific BLR analysis.")
        df_blr = df_blr[~df_blr[category].isin(problem_category_names)]

    # Final check after removing separated categories
    if df_blr.empty or df_blr[category].nunique() < 2 or df_blr['is_fictional'].nunique() < 2:
        print(f"  Skipping {category} after removing separated groups (insufficient data).")
        continue

    # Set the most frequent category as the reference
    ref_category = df_blr[category].mode()[0]
    blr_references[category] = ref_category

    try:
        # Define and fit the model using the new 'is_fictional' variable
        formula = f"is_fictional ~ C({category}, Treatment(reference='{ref_category}'))"
        logit_model = smf.logit(formula, data=df_blr).fit(disp=0) # disp=0 hides convergence messages

        # Calculate Odds Ratios
        odds_ratios_df = pd.DataFrame({
            "Odds Ratio": np.exp(logit_model.params),
            "P-Value": logit_model.pvalues,
            "CI Lower": np.exp(logit_model.conf_int()[0]),
            "CI Upper": np.exp(logit_model.conf_int()[1]),
        })

        print(f"  Reference category for '{category}': {ref_category}")
        print(odds_ratios_df)

        # Write results to a specific sheet in the main Excel file
        with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
            odds_ratios_df.to_excel(writer, sheet_name=f'BLR_Fictional_by_{category[:15]}')

    except Exception as e:
        print(f"  Could not fit model for {category}. Error: {e}")

# After the loop, write the collected details to a new sheet in the Excel file
with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
    # Create DataFrames from the dictionaries, handling unequal list lengths
    other_df_blr = pd.DataFrame({key: pd.Series(value) for key, value in blr_other_details.items()})
    separation_df_blr = pd.DataFrame({key: pd.Series(value) for key, value in blr_separation_details.items()})

    # Write the 'Other' categories table
    pd.DataFrame(["Categories Combined into 'Other'"]).to_excel(writer, sheet_name='BLR_Analysis_Details', index=False, header=False, startrow=0)
    other_df_blr.to_excel(writer, sheet_name='BLR_Analysis_Details', index=False, header=True, startrow=1)

    # Write the 'Separation' categories table below the first one
    start_row_sep = len(other_df_blr) + 3
    pd.DataFrame(["Categories Omitted Due to Separation"]).to_excel(writer, sheet_name='BLR_Analysis_Details', index=False, header=False, startrow=start_row_sep)
    separation_df_blr.to_excel(writer, sheet_name='BLR_Analysis_Details', index=False, header=True, startrow=start_row_sep + 1)

    # Write the reference categories to their own sheet
    references_df_blr = pd.DataFrame.from_dict(blr_references, orient='index', columns=['Reference Category'])
    references_df_blr.index.name = 'Comparison'
    references_df_blr.to_excel(writer, sheet_name="BLR Reference Categories")

## Group Testing
x = [
    'age_rating_category', 'age_rating_descriptions', 'game_modes',
    'genres', 'platforms', 'player_perspectives', 'themes',
    'keywords' # takes a long time to run, can comment out unless interested
]

MIN_CATEGORY_COUNT = 10

kruskal_results = {}
olr_references = {}
olr_other_details = {}
olr_separation_details = {}

for x_header in x:
    for y_header in ['treatment_accuracy', 'recovery_accuracy']:
        print(f"Comparing {x_header} and {y_header}:")

        comparison_name = f"{x_header}-{y_header}"

        # Recovery accuracy analysis requires data with dropped recovery accuracy score of 0
        if y_header == 'recovery_accuracy':
            df_1way = merged_recovery[[x_header, y_header]]
        else:
            df_1way = merged[[x_header, y_header]]

        df_1way = df_1way.explode(x_header)

        df_1way.dropna(subset=[x_header], inplace=True)

        # Skip if not enough data
        if df_1way.empty or df_1way[x_header].nunique() < 2:
            continue

        # Sparsity handling for categories with too few data points
        value_counts = df_1way[x_header].value_counts()
        to_combine = value_counts[value_counts < MIN_CATEGORY_COUNT].index

        if not to_combine.empty:
            olr_other_details[comparison_name] = to_combine.tolist()

        if len(to_combine) > 1:
            print(f"  Combining {len(to_combine)} sparse categories in '{x_header}' into 'Other'.")
            df_1way[x_header] = df_1way[x_header].replace(to_combine, 'Other')

        # Ensure that scores are numbers
        df_1way[y_header] = pd.to_numeric(df_1way[y_header], errors='coerce').astype(int)
        df_1way.dropna(subset=[y_header, x_header], inplace=True)

        # Skip if not enough data now that more stuff has been dropped
        if df_1way.empty or df_1way[x_header].nunique() < 2:
            continue

        # Run a cross-tabulation due to separation
        crosstab_df = pd.crosstab(df_1way[x_header], df_1way[y_header])
        problem_categories = crosstab_df[(crosstab_df > 0).sum(axis=1) == 1]

        if not problem_categories.empty:
            problem_category_names = problem_categories.index
            olr_separation_details[comparison_name] = problem_category_names.tolist()
            print(f"\n  Found {len(problem_category_names)} categories in '{x_header}' causing complete separation: {list(problem_category_names)}")
            print("  Excluding them from this specific OLR analysis.")
            df_1way = df_1way[~df_1way[x_header].isin(problem_category_names)]

        if df_1way.empty or df_1way[x_header].nunique() < 2:
            continue

        formula = f"{y_header} ~ C({x_header})"

        # OLR
        # Calculate reference category
        reference_category = df_1way[x_header].value_counts().index[0]
        olr_references[comparison_name] = reference_category
        formula_olr = f"{y_header} ~ C({x_header}, Treatment(reference='{reference_category}'))"

        # Run OLR
        model_olr = OrderedModel.from_formula(
            formula_olr,
            df_1way,
            distr='logit'
        )

        try:
            res_olr = model_olr.fit(method='bfgs', disp=False)

            olr_summary_df = pd.concat(
                pd.read_html(StringIO(res_olr.summary().as_html()), header=0, index_col=0)
            )

            # Odds ratio for OLR
            olr_summary_df['Odds Ratio'] = np.exp(olr_summary_df['coef'])
            olr_summary_df['OR 2.5%'] = np.exp(olr_summary_df['[0.025'])
            olr_summary_df['OR 97.5%'] = np.exp(olr_summary_df['0.975]'])

            with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
                olr_summary_df.to_excel(writer, sheet_name=f"OLR_{comparison_name[:25]}")

        except Exception as e:
            print(f"  Could not fit OLR model for {comparison_name}. Error: {e}")


        # Kruskal-Wallis
        groups = df_1way.groupby(x_header)[y_header].apply(list)
        kruskal_statistic, p_value = sp.stats.kruskal(*groups)

        # Calculate Eta Squared effect size
        k = len(groups)
        n = len(df_1way)
        eta_squared = (kruskal_statistic - k + 1) / (n - k) if n > k else 0 # Avoid division by zero

        # Store results
        kruskal_results[comparison_name] = {
            'H-statistic': kruskal_statistic,
            'p_value': p_value,
            'eta_squared': eta_squared
        }

        # Significant Kruskal-Walls => Dunn post-hoc
        if (p_value < 0.05):
            dunn_results = posthoc.posthoc_dunn(
                df_1way,
                val_col=y_header,
                group_col=x_header,
                p_adjust='holm'
            )

            with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
                dunn_results.to_excel(writer, sheet_name=f"Dunn_{comparison_name[:24]}")

# Write out end of loop things
kruskal_df = pd.DataFrame.from_dict(kruskal_results, orient='index')
kruskal_df.index.name = 'Comparison'

references_df = pd.DataFrame.from_dict(olr_references, orient='index', columns=['Reference Category'])
references_df.index.name = 'Comparison'

# Create DataFrames for the new OLR details
other_df_olr = pd.DataFrame({key: pd.Series(value) for key, value in olr_other_details.items()})
separation_df_olr = pd.DataFrame({key: pd.Series(value) for key, value in olr_separation_details.items()})

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a', if_sheet_exists='overlay') as writer:
    kruskal_df.to_excel(writer, sheet_name=f"Kruskal-Wallis")
    references_df.to_excel(writer, sheet_name="OLR Reference Categories")

    # Write OLR analysis details to a new sheet
    # 'Other' categories table
    pd.DataFrame(["Categories Combined into 'Other'"]).to_excel(writer, sheet_name='OLR_Analysis_Details', index=False, header=False, startrow=0)
    other_df_olr.to_excel(writer, sheet_name='OLR_Analysis_Details', index=False, header=True, startrow=1)

    # 'Separation' categories table
    start_row_sep_olr = len(other_df_olr) + 3
    pd.DataFrame(["Categories Omitted Due to Separation"]).to_excel(writer, sheet_name='OLR_Analysis_Details', index=False, header=False, startrow=start_row_sep_olr)
    separation_df_olr.to_excel(writer, sheet_name='OLR_Analysis_Details', index=False, header=True, startrow=start_row_sep_olr + 1)
