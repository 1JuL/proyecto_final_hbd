# -*- coding: utf-8 -*-
"""
# Proyecto final Herramientas de Big Data

## Integrantes:
- Diego Norberto Diaz Algarin
- Juan Pablo Moreno Patarroyo
- David Esteban Diaz Vargas
"""

"""# Importaciones"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from prefect import flow, task, get_run_logger

# Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.
from kaggle import KaggleApi

from sqlalchemy import create_engine
from typing import Callable
from functools import wraps
import time

"""# Flow and task definitions"""

DOWNLOAD_DIRECTORY = './data'
TARGET_DATASET = "jul1622/oes-data-converted-format"

if not os.path.exists(DOWNLOAD_DIRECTORY):
    logging.info(f"Creating directory: {DOWNLOAD_DIRECTORY}")
    os.makedirs(DOWNLOAD_DIRECTORY)


def get_directory_size(path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def timing_decorator(func: Callable) -> Callable:
    '''Custom decorator to time task execution'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_run_logger()
        start_time = time.time()

        # Execute the original function
        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(f"Task '{func.__name__}' completed in {execution_time:.2f} seconds")
        return result

    return wrapper

def size_decorator(func: Callable) -> Callable:
    '''Custom decorator to log the size of a directory'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_run_logger()

        directory_size = get_directory_size(DOWNLOAD_DIRECTORY)
        # Execute the original function
        result = func(*args, **kwargs)

        logger.info(f"Downloaded file while executing '{func.__name__}', directory size: {directory_size/2048:.2f} KB ")
        return result

    return wrapper

@task(retries=3, retry_delay_seconds=10, name="Authenticate")
@timing_decorator
def api_authenticate():
    api = KaggleApi()
    api.authenticate()
    return api

@task(retries=3, retry_delay_seconds=10,name='Query Exec')
@timing_decorator
def exec_query(query):
    logger = get_run_logger()

    try:
        db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
        engine = create_engine(db_url)
        df = pd.read_sql_query(query, engine)
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise

    return df

@task(name="Extract Data")
@size_decorator
@timing_decorator
def extract_data():
    logger = get_run_logger()

    try:
        kaggle_api = api_authenticate()
    except Exception as e:
        logger.error(f"Kaggle authentication failed: {e}")
        raise

    logger.info(f"Downloading dataset '{TARGET_DATASET}' to '{DOWNLOAD_DIRECTORY}'...")

    try:
        kaggle_api.dataset_download_files(TARGET_DATASET, path=DOWNLOAD_DIRECTORY, unzip=True)
        logger.info("Dataset downloaded successfully.")
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        raise


@task(name='Transform Data')
@timing_decorator
def transform_data():
    df_m2024 = pd.read_parquet("data/may_2024_data.parquet")
    df_m2023 = pd.read_parquet("data/may_2023_data.parquet")
    df_m2022 = pd.read_parquet("data/may_2022_data.parquet")

    df_may_2024 = df_m2024[["NAICS_TITLE", "OCC_TITLE", "TOT_EMP", "H_MEAN", "A_MEAN"]]
    df_may_2023 = df_m2023[["NAICS_TITLE", "OCC_TITLE", "TOT_EMP", "H_MEAN", "A_MEAN"]]
    df_may_2022 = df_m2022[["NAICS_TITLE", "OCC_TITLE", "TOT_EMP", "H_MEAN", "A_MEAN"]]


    df_2024_clean = df_may_2024.dropna()
    df_2023_clean = df_may_2023.dropna()
    df_2022_clean = df_may_2022.dropna()

    oms_dataframes = [df_2024_clean, df_2023_clean, df_2022_clean]

    return oms_dataframes

@task(retries=3, retry_delay_seconds=10, name='Load Data')
@timing_decorator
def load_data(dfs, table_names=None):
    logger = get_run_logger()
    query = "SELECT version();"

    try:
        db_url = "postgresql+psycopg2://psqluser:psqlpass@localhost:5433/bigdatatools1"
        engine = create_engine(db_url)
        version = exec_query(query)
        logger.info(f"PostgreSQL version: {version.iloc[0,0]}")

        if table_names is None:
            table_names = ["oms_2024", "oms_2023", "oms_2022"]

        for df, name in zip(dfs, table_names):
            logger.info(f"Loading dataframe into table: {name}")
            df.to_sql(name, engine, if_exists='replace', index=False)
    except Exception as e:
        logger.error(f"Error loading data to DB: {e}")
        raise

@flow(name="ETL_Flow")
@timing_decorator
def etl():
    extract_data()
    transformed_dfs = transform_data()
    load_data(transformed_dfs)

etl()

"""# Occupational Employment and Wage Statistics

## Contexto:
El programa de Estadísticas de Empleo y Salarios Ocupacionales (OES) realiza una encuesta semestral para generar estimaciones de empleo y salarios para ocupaciones específicas. El programa OES recopila datos sobre trabajadores asalariados en establecimientos no agrícolas para generar estimaciones de empleo y salarios para aproximadamente 800 ocupaciones. Los datos de trabajadores por cuenta propia no se recopilan y no se incluyen en las estimaciones. El programa OES genera estas estimaciones ocupacionales por área geográfica y por industria. Las estimaciones basadas en áreas geográficas están disponibles a nivel nacional, estatal, metropolitano y no metropolitano. La Oficina de Estadísticas Laborales genera estimaciones de empleo y salarios ocupacionales para más de 450 clasificaciones industriales a nivel nacional. Las clasificaciones industriales corresponden a los grupos industriales del Sistema de Clasificación Industrial de América del Norte (NAICS) de 3, 4 y 5 dígitos.

### Aspectos a tener en cuenta
Antes de realizar la importacion de nuestros datos tuvimos que hacer una limpieza previa para poder convertir nuestros datos en archivo de tipo parquet para mayor portabilidad dado que los archivos xlsx originales tenian un peso elevado dada la cantidad de daots. para hacer esta conversion se tuvieron que reemplazar ciertos simbolos que codificaban ciertos valores:

Notes:
* `*`  = indicates that a wage estimate is not available
* `**`  = indicates that an employment estimate is not available
* `#`  = indicates a wage equal to or greater than `$115.00` per hour or `$239,200` per year
* `~` =indicates that the percent of establishments reporting the occupation is less than `0.5%`


Los astericos de no disponible fueron reemplazados por `None`, el `#` fue reemplazado por `$115.00` en las variables de salarios por horas y `$239,200` fue el valor usado para las variables de salarios anuales por ultimo el simbolo `~` fue reemplazado por 0 ya que en la columna que se usaba para indicar porcentaje este era representado usando numeros enteros de 0-100 y al representar un 0.005 simplemente lo representamos como 0.

Luego de hacer estos arreglos y conversiones extraimos nuestros archivos PARQUET para facilitar el manejo y lectura de este gran volumen de datos.

# Carga de datos
"""

df_m2024 = pd.read_parquet("data/may_2024_data.parquet")
df_m2023 = pd.read_parquet("data/may_2023_data.parquet")
df_m2022 = pd.read_parquet("data/may_2022_data.parquet")

print(df_m2024.shape)
print(df_m2023.shape)
print(df_m2022.shape)

"""## Descripcion de variables

| **Field**          | **Description**                                                                                                       |
| -----------------  | ----------------------------------------------------------------------------------------------------------------------|
| **`area`**         | U.S. (99), state FIPS code, MSA code, or OEWS-specific nonmetropolitan area code                                      |
| **`area_title`**   | Area name                                                                                                             |
| **`area_type`**    | Area type: 1 = U.S.; 2 = State; 3 = U.S. Territory; 4 = Metropolitan Statistical Area (MSA); 6 = Nonmetropolitan Area |
| **`prim_state`**   | The primary state for the given area. "US" is used for the national estimates.                                        |
| **`naics`**        | North American Industry Classification System (NAICS) code for the given industry                                     |
| **`naics_title`**  | NAICS title for the given industry                                                                                    |
| **`i_group`**      | Industry level (cross-industry, NAICS sector, 3–6 digit industry, or OEWS-specific aggregation).                      |
| **`own_code`**     | Ownership type (e.g., 1 = Federal Gov; 2 = State Gov; 3 = Local Gov; 5 = Private; combinations also possible).        |
| **`occ_code`**     | 6-digit SOC (Standard Occupational Classification) or OEWS-specific occupation code                                   |
| **`occ_title`**    | SOC title or OEWS-specific title for the occupation                                                                   |
| **`o_group`**      | SOC occupation level (major, minor, broad, detailed, or OEWS-specific aggregation).                                   |
| **`tot_emp`**      | Estimated total employment (rounded to nearest 10, excludes self-employed).                                           |
| **`emp_prse`**     | Percent relative standard error (PRSE) for employment estimate (sampling error measure).                              |
| **`jobs_1000`**    | Jobs per 1,000 in the given occupation for the area (state/MSA only).                                                 |
| **`loc_quotient`** | Location quotient = occupation’s local employment share ÷ national share (measures concentration).                    |
| **`pct_total`**    | Percent of industry employment in the given occupation (national industry only).                                      |
| **`pct_rpt`**      | Percent of establishments reporting the occupation (national industry only).                                          |
| **`h_mean`**       | Mean hourly wage                                                                                                      |
| **`a_mean`**       | Mean annual wage                                                                                                      |
| **`mean_prse`**    | PRSE for mean wage estimate (sampling error measure).                                                                 |
| **`h_pct10`**      | 10th percentile hourly wage                                                                                           |
| **`h_pct25`**      | 25th percentile hourly wage                                                                                           |
| **`h_median`**     | Median (50th percentile) hourly wage                                                                                  |
| **`h_pct75`**      | 75th percentile hourly wage                                                                                           |
| **`h_pct90`**      | 90th percentile hourly wage                                                                                           |
| **`a_pct10`**      | 10th percentile annual wage                                                                                           |
| **`a_pct25`**      | 25th percentile annual wage                                                                                           |
| **`a_median`**     | Median (50th percentile) annual wage                                                                                  |
| **`a_pct75`**      | 75th percentile annual wage                                                                                           |
| **`a_pct90`**      | 90th percentile annual wage                                                                                           |
| **`annual`**       | “TRUE” if only annual wages are released (e.g., teachers, pilots, athletes).                                          |
| **`hourly`**       | “TRUE” if only hourly wages are released (e.g., actors, dancers, musicians).                                          |

# Manejo de nulos
"""

datasets = {
    "2022": df_m2022,
    "2023": df_m2023,
    "2024": df_m2024
}

# create long dataframe with %
nan_list = []
for year, df in datasets.items():
    nan_perc = df.isnull().mean() * 100
    for col in df.columns:
        if nan_perc[col] > 0:  # keep only variables with NaNs
            nan_list.append({
                "Variable": col,
                "Year": year,
                "NaN_%": round(nan_perc[col], 2)
            })

nan_long = pd.DataFrame(nan_list)

# plot grouped barplot with percentages
plt.figure(figsize=(16, 8))
ax = sns.barplot(data=nan_long, x="Variable", y="NaN_%", hue="Year")

# add % labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", fontsize=8, rotation=60, padding=2)

plt.title("NaN Percentage per Variable by Year")
plt.ylabel("NaN Percentage (%)")
plt.xlabel("Variable")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

"""En esta grafica podemos observar el porcentaje de nulos que tenemos en cada columna en nuestro dataset, como podemos ver hay algunas que tienen valores muy elevados (>20%) pero cuando revisamos que representa cada columna vemos que estas no son relevantes para nuestro analisis.

Las variables de JOBS_100 y LOC_QUOTIENT son indicadores de que tanta concentracion de empleos hay por cada zona, las variables de PCT_TOTAL y PCT_RPT representan el porcentaje de la industria que representa dicha ocupacion, y el PCT_RPT el porcentaje de establecimientos o compañias que reportan tener esa ocupacion y por ultimo las variables de HOURLY y ANNUAL solo es un flag de true o false que nos representa si para cierta ocupacion fueron dadas a conocer las cifras anuales unicamente o por horas unicamente para los salarios.

## Drop nulos
"""

df_m2024.drop(["HOURLY","ANNUAL","PCT_RPT", "PCT_TOTAL","LOC_QUOTIENT","JOBS_1000"],axis=1, inplace= True)
df_m2023.drop(["HOURLY","ANNUAL","PCT_RPT", "PCT_TOTAL","LOC_QUOTIENT","JOBS_1000"],axis=1, inplace= True)
df_m2022.drop(["HOURLY","ANNUAL","PCT_RPT", "PCT_TOTAL","LOC_QUOTIENT","JOBS_1000"],axis=1, inplace= True)

df_2024clean = df_m2024.dropna()
df_2023clean = df_m2023.dropna()
df_2022clean = df_m2022.dropna()

print(df_2024clean.shape)
print(df_2023clean.shape)
print(df_2022clean.shape)

"""### Justificación
Para continuar con nuestro analisis se decidio dropear las columnas "HOURLY" "ANNUAL","PCT_RPT", "PCT_TOTAL","LOC_QUOTIENT","JOBS_1000" ya que estas no nos aportan informacion relevante para nuestro analisis

Al igual que los valores nulos en las columnas ya que estos eran iguales o inferiores al 5%.
"""

datasets = {
    "2022": df_2024clean,
    "2023": df_2023clean,
    "2024": df_2022clean
}

# create long dataframe with %
nan_list = []
for year, df in datasets.items():
    nan_perc = df.isnull().mean() * 100
    for col in df.columns:
      nan_list.append({
        "Variable": col,
        "Year": year,
        "NaN_%": round(nan_perc[col], 2)
      })

nan_long = pd.DataFrame(nan_list)

# plot grouped barplot with percentages
plt.figure(figsize=(17, 4))
ax = sns.barplot(data=nan_long, x="Variable", y="NaN_%", hue="Year")

# add % labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", fontsize=8, rotation=60, padding=2)

plt.title("NaN Percentage per Variable by Year")
plt.ylabel("NaN Percentage (%)")
plt.xlabel("Variable")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

"""# Descripción de variables

## Boxplots for May 2024
"""

# Numeric columns
num_cols = [
    "TOT_EMP", "EMP_PRSE", "H_MEAN", "A_MEAN", "MEAN_PRSE",
    "H_PCT10", "H_PCT25", "H_MEDIAN", "H_PCT75", "H_PCT90",
    "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90"
]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    df_2024clean.boxplot(column=[col], ax=axes[i])
    axes[i].set_title(col, fontsize=10)

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"Boxplots for 2024", fontsize=16, y=1)
plt.tight_layout()
plt.show()

"""### Porcentaje de Outliers por columna 2024"""

def outlier_percentage(df, cols):
    outlier_perc = {}
    for col in cols:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_perc[col] = (len(outliers) / len(series)) * 100
    return pd.Series(outlier_perc, name="Outlier_%")

outlier_summary = outlier_percentage(df_2024clean, num_cols)
print(outlier_summary.round(2))

"""## Boxplots for May 2023

"""

# Numeric columns
num_cols = [
    "TOT_EMP", "EMP_PRSE", "H_MEAN", "A_MEAN", "MEAN_PRSE",
    "H_PCT10", "H_PCT25", "H_MEDIAN", "H_PCT75", "H_PCT90",
    "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90"
]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    df_2023clean.boxplot(column=[col], ax=axes[i])
    axes[i].set_title(col, fontsize=10)

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"Boxplots for 2023", fontsize=16, y=1)
plt.tight_layout()
plt.show()

"""### Porcentaje de Outliers por columna 2023"""

outlier_summary = outlier_percentage(df_2023clean, num_cols)
print(outlier_summary.round(2))

"""## Boxplots for May 2022

"""

# Numeric columns
num_cols = [
    "TOT_EMP", "EMP_PRSE", "H_MEAN", "A_MEAN", "MEAN_PRSE",
    "H_PCT10", "H_PCT25", "H_MEDIAN", "H_PCT75", "H_PCT90",
    "A_PCT10", "A_PCT25", "A_MEDIAN", "A_PCT75", "A_PCT90"
]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    df_2022clean.boxplot(column=[col], ax=axes[i])
    axes[i].set_title(col, fontsize=10)

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f"Boxplots for 2022", fontsize=16, y=1)
plt.tight_layout()
plt.show()

"""### Porcentaje de Outliers por columna 2022"""

outlier_summary = outlier_percentage(df_2022clean, num_cols)
print(outlier_summary.round(2))

"""Con estos boxplots y nuestra tabla de porcetajes podemos observar que tantos datos atipicos hay en cada columna, si observamos unicamente nuestros graficos pareciera que tenemos demasiados datos atipicos pero al ver la cantidad total en porcentaje vemos que la única variable que tiene una cantidad elevada de estos es TOT_EMP (Número total de empleados por cada grupo/area)

Lo cual no es raro de ver ya que hay ciertos empleos o sectores económicos que pueden tener un gran numero de empleados en comparación con los demás al igual que otros sectores los cuales tendrán muy pocos empleados en comparación con las demás áreas laborales

# Analisis
"""

years = [2024, 2023, 2022]

df_list = []

for year in years:
    query = f"""SELECT * FROM oms_{year};"""
    df = exec_query(query)
    df['YEAR'] = year
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)

tech_occupations = [
    'Computer and Information Systems Managers',
    'Software Developers',
    'Data Scientists',
    'Computer Systems Analysts',
    'Information Security Analysts',
    'Computer Programmers',
    'Database Administrators',
    'Software Quality Assurance Analysts and Testers',
    'Web Developers',
    'Software and Web Developers, Programmers, and Testers',
    'Computer Numerically Controlled Tool Programmers',
    'Computer Numerically Controlled Tool Operators and Programmers',
    'Market Research Analysts and Marketing Specialists',
    'Management Analysts',
    'Computer, Automated Teller, and Office Machine Repairers',
    'Computer and Information Research Scientists',
    'Operations Research Analysts',
]

other_occupations = [
    'Farming, Fishing, and Forestry Occupations',
    'Registered Nurses',
    'Pharmacists',
    'Health Education Specialists',
    'Production, Planning, and Expediting Clerks',
    'Mechanical Engineers',
    'Civil Engineers',
    'Industrial Engineers',
    'Engineers, All Other',
    'Health and Safety Engineers, Except Mining Safety Engineers and Inspectors',
    'Career/Technical Education Teachers, Postsecondary',
    'Education Administrators, All Other',
    'Credit Analysts',
    'Electronics Engineers, Except Computer',
    'Chemical Engineers',
    'Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products',
    'Operating Engineers and Other Construction Equipment Operators'
]

# filter to our occupation sets
df_filtered = df_all[df_all['OCC_TITLE'].isin(tech_occupations + other_occupations)].copy()
df_filtered['Group'] = df_filtered['OCC_TITLE'].apply(
    lambda x: 'Tech' if x in tech_occupations else 'Other'
)

# total employment by group per year
emp_by_group = df_filtered.groupby(['YEAR','Group'])['TOT_EMP'].sum().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(x='YEAR', y='TOT_EMP', hue='Group', data=emp_by_group)
plt.title("Total Employment: Tech vs Other Occupations")
plt.ylabel("Employment (Total)")
plt.tight_layout()
plt.show()

# mean annual wage by group per year
wage_by_group = df_filtered.groupby(['YEAR','Group'])['A_MEAN'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(x='YEAR', y='A_MEAN', hue='Group', data=wage_by_group)
plt.title("Average Annual Mean Wage: Tech vs Other Occupations")
plt.ylabel("Annual Mean Wage (USD)")
plt.tight_layout()
plt.show()

# wage distributions per group
plt.figure(figsize=(10,6))
sns.boxplot(x='YEAR', y='A_MEAN', hue='Group', data=df_filtered)
plt.title("Distribution of Annual Mean Wages (A_MEAN)")
plt.ylabel("Annual Mean Wage (USD)")
plt.yscale("log")
plt.tight_layout()
plt.show()

# line plot of employment trends per occupation
plt.figure(figsize=(16,10))
sns.lineplot(
    data=df_filtered, x='YEAR', y='TOT_EMP', hue='OCC_TITLE', marker='o'
)
plt.title("Employment Trend per Occupation (2022–2024)")
plt.ylabel("Total Employment")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# heatmap of mean wages per occupation and year

wage_pivot = df_filtered.pivot_table(
    index='OCC_TITLE', columns='YEAR', values='A_MEAN'
)

plt.figure(figsize=(10,8))
sns.heatmap(wage_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Annual Mean Wages by Occupation and Year")
plt.ylabel("Occupation")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# scatter plot of wages vs employment

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_filtered, x='A_MEAN', y='TOT_EMP', hue='Group', style='YEAR', s=100
)
plt.title("Wages vs Employment by Occupation and Year")
plt.xlabel("Annual Mean Wage (USD)")
plt.ylabel("Employment")
plt.tight_layout()
plt.show()

# plot of wage gap over time

wage_gap = df_filtered.groupby(['YEAR','Group'])['A_MEAN'].mean().unstack()

plt.figure(figsize=(8,5))
(wage_gap['Tech'] - wage_gap['Other']).plot(kind='bar')
plt.title("Tech vs Other Wage Gap")
plt.ylabel("Difference in Annual Mean Wage (USD)")
plt.tight_layout()
plt.show()

# line plot of employment trends per tech occupation

plt.figure(figsize=(16,10))
sns.lineplot(
    data=df_filtered[df_filtered['Group'] == 'Tech'], x='YEAR', y='TOT_EMP', hue='OCC_TITLE', marker='o'
)
plt.title("Employment Trend per Tech Occupation (2022–2024)")
plt.ylabel("Total Employment")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()