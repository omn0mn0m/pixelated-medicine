# Code
This project uses Python for statistical analysis and generating files for the research project. It is set up to use `uv`, but can be run using default Python tooling as well.

## Statistical Analysis
### .env
```
IGDB_CLIENT=
IGDB_TOKEN=
```

### Dependencies
The statistical analysis for this project depends on a few stats packages.

#### Install via Pip
```
pip install .
```

#### Install via uv
You do not need to pre-install dependencies with uv, but if you want to anyways, you can use:

```
uv sync
```

### Run Analysis
There is a list `REGRESSION_CATEGORIES` where you should comment or uncomment `keywords` depending on if you want to analyze the `keywords` game characteristic. This characteristic has hundreds of values, so it can take a while to run. It is uncommented by default so that the full analysis used for the paper is run.

#### Using Python
```
python main.py
```

#### Using uv
```
uv run main.py
```

## Scoring Rubric Validation
### Creating Validation Reviewer File
To validate the scoring rubric used for treatment and recovery accuracy, an third-party reviewer was recruited to score a random selection of medical encounters. Intraclass correlation was then calculated.

You can create the file for the third-party reviewer to enter data with:

```
uv run icc_data_selection.py
```

or 

```
python icc_data_selection.py
```

### Run ICC Analysis
```
uv run icc_analyze.py
```

or 

```
python icc_analyze.py
```
