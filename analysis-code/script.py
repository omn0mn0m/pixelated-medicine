import os
import json
from enum import Enum

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from igdb.wrapper import IGDBWrapper
import matplotlib.pyplot as plt
import requests

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sp

load_dotenv()

treatment_accuracy_conversion = {
    "0 - Completely Fictional/ Unable to Assess": 0,
    "1 - Completely Inaccurate": 1,
    "2 - Mostly Inaccurate": 2,
    "3 - Half Accurate": 3,
    "4 - Mostly Accurate": 4,
    "5 - Completely Accurate": 5,
}

recovery_accuracy_conversion = {
    "0 - Recovery time not assessible": 0,
    "1 - Recovery time too short": 1,
    "2 - Recovery time accurate": 2,
    "3 - Recovery time too long": 3,
}

class AgeRatingCategory(Enum):
    AGERATING_CATEGORY_NULL = 0
    ESRB = 1
    PEGI = 2
    CERO = 3
    USK = 4
    GRAC = 5
    CLASS_IND = 6
    ACB = 7

class AgeRatingRating(Enum):
    AGERATING_RATING_NULL = 0
    THREE = 1
    SEVEN = 2
    TWELVE = 3
    SIXTEEN = 4
    EIGHTEEN = 5
    RP = 6
    EC = 7
    E = 8
    E10 = 9
    T = 10
    M = 11
    AO = 12
    CERO_A = 13
    CERO_B = 14
    CERO_C = 15
    CERO_D = 16
    CERO_Z = 17
    USK_0 = 18
    USK_6 = 19
    USK_12 = 20
    USK_16 = 21
    USK_18 = 22
    GRAC_ALL = 23
    GRAC_TWELVE = 24
    GRAC_FIFTEEN = 25
    GRAC_EIGHTEEN = 26
    GRAC_TESTING = 27
    CLASS_IND_L = 28
    CLASS_IND_TEN = 29
    CLASS_IND_TWELVE = 30
    CLASS_IND_FOURTEEN = 31
    CLASS_IND_SIXTEEN = 32
    CLASS_IND_EIGHTEEN = 33
    ACB_G = 34
    ACB_PG = 35
    ACB_M = 36
    ACB_MA15 = 37
    ACB_R18 = 38
    ACB_RC = 39

class AgeRatingContentDescriptionCategory(Enum):
    AGERATINGCONTENTDESCRIPTION_CATEGORY_NULL = 0
    ESRB_ALCOHOL_REFERENCE = 1
    ESRB_ANIMATED_BLOOD = 2
    ESRB_BLOOD = 3
    ESRB_BLOOD_AND_GORE = 4
    ESRB_CARTOON_VIOLENCE = 5
    ESRB_COMIC_MISCHIEF = 6
    ESRB_CRUDE_HUMOR = 7
    ESRB_DRUG_REFERENCE = 8
    ESRB_FANTASY_VIOLENCE = 9
    ESRB_INTENSE_VIOLENCE = 10
    ESRB_LANGUAGE = 11
    ESRB_LYRICS = 12
    ESRB_MATURE_HUMOR = 13
    ESRB_NUDITY = 14
    ESRB_PARTIAL_NUDITY = 15
    ESRB_REAL_GAMBLING = 16
    ESRB_SEXUAL_CONTENT = 17
    ESRB_SEXUAL_THEMES = 18
    ESRB_SEXUAL_VIOLENCE = 19
    ESRB_SIMULATED_GAMBLING = 20
    ESRB_STRONG_LANGUAGE = 21
    ESRB_STRONG_LYRICS = 22
    ESRB_STRONG_SEXUAL_CONTENT = 23
    ESRB_SUGGESTIVE_THEMES = 24
    ESRB_TOBACCO_REFERENCE = 25
    ESRB_USE_OF_ALCOHOL = 26
    ESRB_USE_OF_DRUGS = 27
    ESRB_USE_OF_TOBACCO = 28
    ESRB_VIOLENCE = 29
    ESRB_VIOLENT_REFERENCES = 30
    ESRB_ANIMATED_VIOLENCE = 31
    ESRB_MILD_LANGUAGE = 32
    ESRB_MILD_VIOLENCE = 33
    ESRB_USE_OF_DRUGS_AND_ALCOHOL = 34
    ESRB_DRUG_AND_ALCOHOL_REFERENCE = 35
    ESRB_MILD_SUGGESTIVE_THEMES = 36
    ESRB_MILD_CARTOON_VIOLENCE = 37
    ESRB_MILD_BLOOD = 38
    ESRB_REALISTIC_BLOOD_AND_GORE = 39
    ESRB_REALISTIC_VIOLENCE = 40
    ESRB_ALCOHOL_AND_TOBACCO_REFERENCE = 41
    ESRB_MATURE_SEXUAL_THEMES = 42
    ESRB_MILD_ANIMATED_VIOLENCE = 43
    ESRB_MILD_SEXUAL_THEMES = 44
    ESRB_USE_OF_ALCOHOL_AND_TOBACCO = 45
    ESRB_ANIMATED_BLOOD_AND_GORE = 46
    ESRB_MILD_FANTASY_VIOLENCE = 47
    ESRB_MILD_LYRICS = 48
    ESRB_REALISTIC_BLOOD = 49
    PEGI_VIOLENCE = 50
    PEGI_SEX = 51
    PEGI_DRUGS = 52
    PEGI_FEAR = 53
    PEGI_DISCRIMINATION = 54
    PEGI_BAD_LANGUAGE = 55
    PEGI_GAMBLING = 56
    PEGI_ONLINE_GAMEPLAY = 57
    PEGI_IN_GAME_PURCHASES = 58
    CERO_LOVE = 59
    CERO_SEXUAL_CONTENT = 60
    CERO_VIOLENCE = 61
    CERO_HORROR = 62
    CERO_DRINKING_SMOKING = 63
    CERO_GAMBLING = 64
    CERO_CRIME = 65
    CERO_CONTROLLED_SUBSTANCES = 66
    CERO_LANGUAGES_AND_OTHERS = 67
    GRAC_SEXUALITY = 68
    GRAC_VIOLENCE = 69
    GRAC_FEAR_HORROR_THREATENING = 70
    GRAC_LANGUAGE = 71
    GRAC_ALCOHOL_TOBACCO_DRUG = 72
    GRAC_CRIME_ANTI_SOCIAL = 73
    GRAC_GAMBLING = 74
    CLASS_IND_VIOLENCIA = 75
    CLASS_IND_VIOLENCIA_EXTREMA = 76
    CLASS_IND_CONTEUDO_SEXUAL = 77
    CLASS_IND_NUDEZ = 78
    CLASS_IND_SEXO = 79
    CLASS_IND_SEXO_EXPLICITO = 80
    CLASS_IND_DROGAS = 81
    CLASS_IND_DROGAS_LICITAS = 82
    CLASS_IND_DROGAS_ILICITAS = 83
    CLASS_IND_LINGUAGEM_IMPROPRIA = 84
    CLASS_IND_ATOS_CRIMINOSOS = 85

access_token = requests.post('https://id.twitch.tv/oauth2/token?client_id={}&client_secret={}&grant_type=client_credentials'.format(os.getenv('IGDB_CLIENT', 'oops'), os.getenv('IGDB_TOKEN', 'oops'))).json()['access_token']

wrapper = IGDBWrapper(os.getenv('IGDB_CLIENT', 'oops'), access_token)

# Load and scrub data
injuries = pd.read_excel('Injury Data Collection.xlsx')
pd.set_option('future.no_silent_downcasting', True)
injuries = injuries.replace({"Treatment Accuracy": treatment_accuracy_conversion,
                  "Recovery Accuracy": recovery_accuracy_conversion,
                  })
injuries.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_'), inplace=True)
injuries['icd_10'] = injuries['icd_10'].str.split('.').str[0]
injuries['icd_10'] = injuries['icd_10'].str.split(',')
print(injuries)
print(injuries)
print(injuries["icd_10"].explode().value_counts()[0:10])

# Get game data
unique_games = injuries['igdb_id'].unique()

game_info = []

for game in unique_games:
    search_string = ('fields name,age_ratings.category,age_ratings.rating,'
                     'age_ratings.content_descriptions.category,aggregated_rating,category,'
                     'franchise.name,game_engines.name,game_modes.name,genres.name,'
                     'involved_companies.company.name,keywords.name,multiplayer_modes,first_release_date,'
                     'platforms.name,player_perspectives.name,storyline,summary,themes.name; '
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
    age_ratings = []
    
    if not isinstance(ratings, list):
        return

    for rating in ratings:
        cleaned_rating = {
            'category': AgeRatingCategory(rating['category']),
            'rating': AgeRatingRating(rating['rating']),
        }
        
        if 'content_descriptions' in rating:
            cleaned_rating['content_descriptions'] = [AgeRatingContentDescriptionCategory(description['category']) for description in rating['content_descriptions']]

        age_ratings.append(cleaned_rating)
                
    return age_ratings

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

print(games)

# Stats
injuries_descriptive = injuries.describe(include='all')
print(injuries_descriptive)

games_descriptive = games.describe(include='all')
print(games_descriptive)

for category in ['age_ratings', 'franchise', 'game_modes', 'genres', 'involved_companies', 'keywords', 
                 'platforms', 'player_perspectives', 'themes', 'game_engines']:
    print(games[category].explode(category).describe(include='all'))

merged = injuries.merge(games, left_on='igdb_id', right_on='id')
print(merged)

merged_descriptive = merged.describe(include='all')
print(merged_descriptive)

for category in ['age_ratings', 'franchise', 'game', 'game_modes', 'genres', 'involved_companies', 'keywords', 
                 'platforms', 'player_perspectives', 'themes', 'game_engines']:
    print(merged[category].explode(category).describe(include='all'))

## Means
for category in ['genres', 'themes']:
    print(merged.explode(category).groupby(category)[['treatment_accuracy']].mean())

## Pearson correlation
correlation = merged[['treatment_accuracy', 'recovery_accuracy']].corr()
pearsonr = sp.stats.pearsonr(merged['treatment_accuracy'].astype(int), merged['recovery_accuracy'].astype(int))

print(correlation)
print(pearsonr)

## ANOVA
y = ['treatment_accuracy', 'recovery_accuracy']
x = ['game_modes', 'genres', 'keywords', 'platforms', 'player_perspectives', 'themes']

for x_header in x:
    for y_header in y:
        df_1way = merged[[x_header, y_header]]
        df_1way = df_1way.explode(x_header)
        X = df_1way[x_header].astype('category')
        Y = df_1way[y_header].astype(int)

        model = smf.ols(formula=f'Y ~ C(X)', data=df_1way).fit()
        print(model.summary())

        anova_table = sm.stats.anova_lm(model, type=2)
        print(anova_table)

        multi = sm.stats.multicomp.MultiComparison(np.array(df_1way[y_header], dtype='float64'), df_1way[x_header])
        kruskal = multi.kruskal()
        print(kruskal)
        tukey = multi.tukeyhsd()
        print(tukey.summary())

        input()

# Write out
with pd.ExcelWriter("Injuries.xlsx") as writer:
    injuries.to_excel(writer, sheet_name='Cleaned')
    injuries_descriptive.to_excel(writer, sheet_name='Descriptive')

with pd.ExcelWriter("Games.xlsx") as writer:
    games.to_excel(writer, sheet_name='Cleaned')
    games_descriptive.to_excel(writer, sheet_name='Descriptive')

with pd.ExcelWriter("Merged.xlsx") as writer:
    merged.to_excel(writer, sheet_name='Merged')
    merged_descriptive.to_excel(writer, sheet_name='Descriptive')
