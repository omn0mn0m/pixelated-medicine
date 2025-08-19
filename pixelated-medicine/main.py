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

## Means for categories that need exploding
for category in ['age_rating_descriptions', 'genres', 'themes']:
    exploded_merged = merged.explode(category)
    exploded_merged_recovery = merged_recovery.explode(category)
    
    mean_exploded = exploded_merged.groupby(category)[['treatment_accuracy']].mean()
    mean_rec_exploded = exploded_merged_recovery.groupby(category)[['recovery_accuracy']].mean()

    with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
        mean_exploded.to_excel(writer, sheet_name=f"Treatment Mean_{category}")
        mean_rec_exploded.to_excel(writer, sheet_name=f"Recovery Mean_{category}")

## Correlation - Spearman's chosen due to ordinal data
data_to_corr = pd.DataFrame({
    'treatment_accuracy': merged_recovery['treatment_accuracy'].astype(int),
    'reecovery_accuracy': merged_recovery['recovery_accuracy'].astype(int),
    'total_rating': merged_recovery['total_rating']
})

spearman_matrix = data_to_corr.corr(method='spearman')
print(spearman_matrix)

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
    spearman_matrix.to_excel(writer, sheet_name=f"Spearman's R")

## Group Testing
y = ['treatment_accuracy', 'recovery_accuracy']
x = ['age_rating_category', 'age_rating_descriptions', 'game_modes', 'genres', 'keywords', 'platforms', 'player_perspectives', 'themes']

shapiro_results = {}
kruskal_results = {}

for x_header in x:
    for y_header in y:
        print(f"Comparing {x_header} and {y_header}:")

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
        
        df_1way[y_header] = pd.to_numeric(df_1way[y_header], errors='coerce')
        df_1way.dropna(subset=[y_header, x_header], inplace=True)
        df_1way[y_header] = df_1way[y_header].astype(int)

        if df_1way.empty or df_1way[x_header].nunique() < 2:
            continue

        # Skip if not enough data now that more stuff has been dropped
        if df_1way.empty or df_1way[x_header].nunique() < 2:
            continue

        formula = f"{y_header} ~ C({x_header})"

        # OLS
        res_ols = smf.ols(formula=formula, data=df_1way).fit()
        ols_summary_tables = pd.read_html(StringIO(res_ols.summary().as_html()), header=0, index_col=0)
        ols_summary_df = pd.concat(ols_summary_tables)

        with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
            ols_summary_df.to_excel(writer, sheet_name=f"OLS_{x_header}-{y_header}")

        # OLR
        model_olr = OrderedModel.from_formula(
            formula,
            df_1way,
            distr='logit'
        )
        res_olr = model_olr.fit(method='bfgs', disp=False)
        
        olr_summary_tables = pd.read_html(StringIO(res_olr.summary().as_html()), header=0, index_col=0)
        olr_summary_df = pd.concat(olr_summary_tables)
        with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
            olr_summary_df.to_excel(writer, sheet_name=f"OLR_{x_header}-{y_header}")

        # Shapiro-Wilk
        shapiro_results[f"{x_header}-{y_header}"] = sp.stats.shapiro(res_ols.resid)

        # Non-parametric => run  Kruskal-Wallis
        if shapiro_results[f"{x_header}-{y_header}"].pvalue <= 0.05:
            multi = sm.stats.multicomp.MultiComparison(
                np.array(df_1way[y_header], dtype='float64'),
                df_1way[x_header]
            )
            kruskal = multi.kruskal()
            kruskal_results[f"{x_header}-{y_header}"] = kruskal

            # Significant Kruskal-Walls => Dunn post-hoc
            if (kruskal < 0.05):
                dunn_results = posthoc.posthoc_dunn(
                    df_1way,
                    val_col=y_header,
                    group_col=x_header,
                    p_adjust='holm'
                )
                print(dunn_results)

                with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
                    dunn_results.to_excel(writer, sheet_name=f"Dunn_{x_header}-{y_header}")
        # Parametric => run ANOVA
        else:
            anova_df = sm.stats.anova_lm(res_ols, type=2)
            print(anova_df)

            with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
                anova_df.to_excel(writer, sheet_name=f"ANOVA_{x_header}-{y_header}")

            # Significant ANOVA => Tukey post-hoc
            if anova_df['PR(>F)'][0] < 0.05:
                multi = sm.stats.multicomp.MultiComparison(
                    np.array(df_1way[y_header], dtype='float64'),
                    df_1way[x_header]
                )
                
                tukey = multi.tukeyhsd()
                tukey_df = pd.DataFrame(tukey._results_table.data[1:],
                                        columns=tukey._results_table.data[0])
                tukey_df['variable'] = y_header
                tukey_df['grouping'] = x_header

# Write out the Shapiro-Wilk results
shapiro_df = pd.DataFrame.from_dict(shapiro_results, orient='index', columns=['statistic', 'pvalue'])
shapiro_df.index.name = 'Comparison'

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
    shapiro_df.to_excel(writer, sheet_name=f"Shapiro-Wilk")

# Write out Kruskal-Wallis results
kruskal_df = pd.DataFrame.from_dict(kruskal_results, orient='index', columns=['p_value'])
kruskal_df.index_Name = 'Comparison'

with pd.ExcelWriter("./stats/Merged.xlsx", mode='a') as writer:
    kruskal_df.to_excel(writer, sheet_name=f"Kruskal-Wallis")
