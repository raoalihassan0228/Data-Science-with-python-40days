import pandas as pd
import pandas_profiling
import streamlit as st
import seaborn as sns

from streamlit_pandas_profiling import st_profile_report

df = sns.load_dataset('iris')

# applying profiling report function on dataset
pr = df.profile_report()
# run in streamlit app
st_profile_report(pr)