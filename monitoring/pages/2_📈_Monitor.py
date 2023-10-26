import streamlit as st

import pandas as pd
import pymonetdb
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title('Options')

conn = pymonetdb.connect(database='dataflow_analyzer', port=50000, username='monetdb', password='monetdb')
cursor = conn.cursor()

# Create a function to retrieve data from the database
def fetch_data(table = 'oservertrainingaggregation', server_id=0):
    cursor.execute(f'SELECT * FROM {table} WHERE server_id={server_id}')  # Modify to your table name and SQL query

    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=columns)
    return df

tables = [
    "oServerTrainingAggregation",
    "oClientTraining",
    "iTrainingConfig",
    "oTrainingConfig",
    "iEvaluationConfig",
    "oClientEvaluation",
    "oServerEvaluationAggregation"
]


# Allow the user to select which attributes to plot
st.sidebar.subheader('Select Attributes to Plot')

selected_table = st.sidebar.selectbox('Select table:', tables, index=0)

def get_server_ids():
    cursor.execute(f'SELECT DISTINCT server_id FROM iserverconfig')
    data = cursor.fetchall()
    server_ids = [int(row[0]) for row in data]
    return server_ids

selected_server_id = st.sidebar.selectbox('Select server ID:', get_server_ids())
data_df = fetch_data(selected_table, selected_server_id)


selected_column = st.sidebar.selectbox('Select column:', data_df.columns, index=8)
sns.set_style("darkgrid")
if selected_column:
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title(f"Evolution of the {' '.join(selected_column.split('_'))} over training", fontsize=25)
    sns.lineplot(x=data_df["server_round"], y=data_df[selected_column],  linestyle=':', marker="o")
    plt.xlabel('Round', fontsize=15)
    ax.xaxis.get_major_locator().set_params(integer=True)    
    plt.ylabel(f'{selected_column.capitalize()}', fontsize=15)
    st.pyplot(fig)

conn.close()