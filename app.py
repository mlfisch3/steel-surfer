import numpy as np
import pandas as pd
import streamlit as st

np.set_printoptions(precision=2, suppress=True)
pd.set_option("precision", 2)
st.set_page_config(layout="wide")

def run_app(fpath):

    @st.cache
    def load_steel_data(fpath):
        
        data = pd.read_csv(fpath, sep='|', low_memory=False)
        data.sort_values(['Steel'], ignore_index=True, inplace=True)
        property_columns = ['Steel', 'C', 'Mn', 'Cr', 'Ni', 'V', 'Mo', 'W', 'Co', 'Si', 'S', 'P', 'Rw', 'Type', 'cluster']
        data_properties = data[property_columns]
        
        return data, data_properties

    @st.cache
    def get_steel_cluster(df,steel):

        if np.array(df['Steel']==steel, dtype=int).sum() == 0:
            return pd.DataFrame({'Steel':'','Rw':'','-':''}, index=[])
        
        cluster = df[df['Steel']==steel]['cluster'].iloc[0]
        dist = 'd' + str(cluster)

        return df[df['cluster']==cluster][['Steel','Rw',dist]].sort_values([dist], ignore_index=True) 


    steel_data, steel_properties = load_steel_data(fpath)
    
    steel_names = [x for x in sorted(steel_data['Steel'].tolist())]
    steel_table_selection = st.sidebar.multiselect('Choose steels:', steel_names, default=steel_names)
    steel_properties_subset = steel_properties[steel_properties['Steel'].isin(steel_table_selection)]
    st.table(steel_properties_subset)

    steel_selection = st.sidebar.selectbox("View steels similar to: ", steel_names)
    steel_cluster = get_steel_cluster(steel_data, steel_selection)
    steel_cluster.columns = ['Steel', 'Hardness', 'distance']
    steel_cluster.sort_values(['distance'], ignore_index=True, inplace=True)
    st.sidebar.table(steel_cluster)

if __name__ == '__main__':

    data_file_path = "steels_data_clusters15.psv"

    run_app(data_file_path)