import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')

# import data
data = pd.read_csv("../data/dataset.csv")

print(data.shape)
print(data.head())
print(data.nunique())
print(data.info(verbose=True, show_counts=True))
print(data.describe())

# Removes an empty feature
data = data.drop(['Unnamed: 83'], axis=1)

print(data.shape)
print("No. of rows with missing values:", data.isnull().any(axis=1).sum())
print(data.columns)

# Removes features not usefull
data.drop(['encounter_id', 'patient_id', 'icu_admit_source',
           'icu_id', 'icu_stay_type', 'icu_type'], axis=1, inplace=True)

# Removes null values in those features
data = data[data[['bmi', 'weight', 'height']].isna().sum(axis=1) == 0]

############################################### Plots ##############################################
fig = px.histogram(data[['age', 'gender', 'hospital_death', 'bmi']].dropna(), x="age", y="hospital_death",
                   color='gender',
                   marginal='box', hover_data=data[['age', 'gender', 'hospital_death', 'bmi']].columns)
fig.show()

unpivot = pd.melt(data, data.describe().columns[0], data.describe().columns[1:])

fig2 = sns.FacetGrid(unpivot, col='variable', col_wrap=3,
                     sharex=False, sharey=False)
fig2.map(sns.kdeplot, "value")
plt.show()

weight_data = data[['weight', 'hospital_death', 'bmi']]
weight_data['weight'] = weight_data['weight'].round(0)
weight_data['bmi'] = weight_data['bmi'].round(0)
weight_death = weight_data[['weight', 'hospital_death']].groupby('weight').mean().reset_index()
bmi_death = weight_data[['bmi', 'hospital_death']].groupby('bmi').mean().reset_index()
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.add_trace(
    go.Scatter(x=weight_death['weight'], y=weight_death['hospital_death'], name="Weight"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=bmi_death['bmi'], y=bmi_death['hospital_death'], name="BMI"),
    row=1, col=2
)
fig.update_layout(
    title_text="<b>impacts of BMI and weight over patients<b>"
)
fig.update_yaxes(title_text="<b>Average Hospital Death")
fig.show()