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

apache3 = data[['age', 'apache_3j_bodysystem', 'hospital_death']]
apache3 = apache3.groupby(['apache_3j_bodysystem', 'age']).agg(['size', 'mean']).reset_index()

apache3['size'] = apache3['hospital_death']['size']
apache3['mean'] = apache3['hospital_death']['mean']

apache3.drop('hospital_death', axis=1, inplace=True)

systems = list(apache3['apache_3j_bodysystem'].unique())
data = []
list_updatemenus = []
for n, s in enumerate(systems):
    visible = [False] * len(systems)
    visible[n] = True
    temp_dict = dict(label=str(s),
                     method='update',
                     args=[{'visible': visible},
                           {'title': '<b>' + s + '<b>'}])
    list_updatemenus.append(temp_dict)

for s in systems:
    mask = (apache3['apache_3j_bodysystem'].values == s)
    trace = (dict(visible=False,
                  x=apache3.loc[mask, 'age'],
                  y=apache3.loc[mask, 'mean'],
                  mode='markers',
                  marker={'size': apache3.loc[mask, 'size'] / apache3.loc[mask, 'size'].sum() * 1000,
                          'color': apache3.loc[mask, 'mean'],
                          'showscale': True})
             )
    data.append(trace)

data[0]['visible'] = True

layout = dict(updatemenus=list([dict(buttons=list_updatemenus)]),
              xaxis=dict(title='<b>Age<b>', range=[min(apache3.loc[:, 'age']) - 10, max(apache3.loc[:, 'age']) + 10]),
              yaxis=dict(title='<b>Average Hospital Death<b>',
                         range=[min(apache3.loc[:, 'mean']) - 0.1, max(apache3.loc[:, 'mean']) + 0.1]),
              title='<b>Survival Rate<b>')
fig = dict(data=data, layout=layout)
fig.show()
