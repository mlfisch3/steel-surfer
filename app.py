import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
from scipy.spatial import distance   

title = f'Steel Surfer'
st.set_page_config(page_title=title, layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

np.set_printoptions(precision=2, suppress=True)
pd.set_option("display.precision", 2)
pd.set_option("styler.format.precision", 2)
pd.options.display.colheader_justify='center'

# # TODO: Add distance heat map

from scipy.spatial import distance

def dist(x,i,j):
    a = np.array(x.iloc[i].to_list())
    b = np.array(x.iloc[j].to_list())
    r = a-b
    return np.sqrt(np.dot(r,r))

def heat(x, measure='euclidean'):
    N = len(x)
    x_num = x[x.describe().columns]
    d = distance.pdist(x_num,measure)
    dist_map = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            dist_map[i,j] = dist(x_num, i,j)
            
    return dist_map

###########
#
#   ...
# dist_map = heat()
# sns.heatmap(dist_map)


def run_app(fpath):

    st.write(st.session_state.x)
    #@st.cache
    def load_steel_data(fpath):
        
        data = pd.read_csv(fpath, sep='|', low_memory=False)
        data.sort_values(['Steel'], ignore_index=True, inplace=True)
        property_columns = ['Steel', 'C', 'Mn', 'Cr', 'Ni', 'V', 'Mo', 'W', 'Co', 'Si', 'S', 'P', 'Rw', 'Type', 'cluster']
        data_properties = data[property_columns]
        
        return data, data_properties

    #@st.cache
    def get_steel_cluster(df,steel):#, use_container_width=True):  use_container_width only since version 1.13

        if np.array(df['Steel']==steel, dtype=int).sum() == 0:
            return pd.DataFrame({'Steel':'','Rw':'','-':''}, index=[])
        
        cluster = df[df['Steel']==steel]['cluster'].iloc[0]
        dist = 'd' + str(cluster)

        return df[df['cluster']==cluster][['Steel','Rw',dist]].sort_values([dist], ignore_index=True) 


    steel_data, steel_properties = load_steel_data(fpath)
    
    steel_names = [x for x in sorted(steel_data['Steel'].tolist())]

    with st.sidebar:

        with st.expander("Steels"):
            steel_table_selection = st.multiselect('Choose steels:', steel_names, default=steel_names)    
            steel_properties_subset = steel_properties[steel_properties['Steel'].isin(steel_table_selection)]

        steel_selection = st.selectbox("View steels similar to: ", steel_names)
        steel_cluster = get_steel_cluster(steel_data, steel_selection)
        steel_cluster.columns = ['Steel', 'Hardness', 'distance']
        steel_cluster.sort_values(['distance'], ignore_index=True, inplace=True)
        st.sidebar.table(steel_cluster.style.background_gradient(cmap='cividis'))#, use_container_width=use_container_width)

    st.markdown("<h1 style='text-align: justify; color: white;'>Steel Chart</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: justify; color: white;'>Elemental Composition (%) & Rockwell Hardness (Rw)</h3>", unsafe_allow_html=True)

    st.dataframe(steel_properties_subset.style.background_gradient(cmap='bone'))#, use_container_width=use_container_width)
    
    steel_properties_cluster = steel_properties[steel_properties['Steel'].isin(steel_cluster['Steel'])]

    #d = distance.pdist(composition_df, 'euclidean')
    #st.dataframe(steel_properties_cluster.style.background_gradient(cmap='bone'))
    st.dataframe(steel_properties_cluster.style.background_gradient(cmap='bone'))#, use_container_width=use_container_width)
    # N = len(steel_cluster['Steel'])

    composition_columns = ['C', 'Mn', 'Cr', 'Ni', 'V', 'Mo', 'W', 'Co', 'Si', 'S', 'P']
    cluster_composition_df = steel_properties_cluster[composition_columns]
    # steel_names_abbr = steel_names[0:N]
    cluster_dist_map = heat(cluster_composition_df)
    #dist_map_df = pd.DataFrame(data=dist_map, index=steel_names_abbr, columns=steel_names_abbr)
    cluster_dist_map_df = pd.DataFrame(data=cluster_dist_map+cluster_dist_map.T, index=steel_cluster['Steel'], columns=steel_cluster['Steel'])
    st.dataframe(cluster_dist_map_df.style.background_gradient(cmap='bone'))#, use_container_width=use_container_width)
    fig, ax = plt.subplots()    
    sns.heatmap(cluster_dist_map_df, ax=ax)
    st.write(fig)

if __name__ == '__main__':


    if 'x' not in st.session_state:
        st.session_state.x = ''
        
    data_file_path = "steels_data_clusters15.psv"

    run_app(data_file_path)