import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.write('''
# GDP growth App
## Created by Rao Ali Hassan         
Streamlit web app
''')

# plotly animated plots
df1 = px.data.gapminder()
fig2 = px.scatter(df1, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
st.plotly_chart(fig2)