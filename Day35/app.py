import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.write('''
# Random Forest Classifier App
## Created by Rao Ali Hassan         
This app predicts the type of iris based on sepal length, sepal widh, petal length, and petal width.
''')

st.sidebar.header("Change IRIS parameter")

# defining a function
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal lenth", 4.3, 15.0, 5.5)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features
# call the function
df = user_input_features()

st.subheader("IRIS parameters")
st.write(df)

iris = sns.load_dataset('iris')
st.subheader('Iris datadet')
st.write(iris.head(10))

# plotly plots
st.subheader('Plotly plot')
fig = px.box(iris, x='species', y='petal_length', color='species')
st.plotly_chart(fig)

# # seaborn plots
# st.subheader('seaborn plot')
# sns.barplot(x=iris['species'], y=iris['petal_length'])
# st.pyplot()

X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']

model = RandomForestClassifier()
model.fit(X,y)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
st.subheader('Class labels and their corresponding index number')
st.write(iris['species'].unique())

st.subheader('Prediction')
p=st.write(prediction[0])
st.write(p)

# prediction chances
st.markdown('### Prediction chances/probability')
st.write(prediction_proba)


# plotly animated plots
df1 = px.data.gapminder()
fig2 = px.scatter(df1, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
st.plotly_chart(fig2)