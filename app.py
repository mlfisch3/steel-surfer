import numpy as np
import pandas as pd
import pandas.io.formats.style
import streamlit as st
from scipy.spatial import distance   #

np.set_printoptions(precision=2, suppress=True)
pd.set_option("precision", 2)
pd.options.display.colheader_justify='center'
st.set_page_config(layout="wide")



# # TODO: Add distance heat map

# from scipy.spatial import distance

# def dist(x,i,j):
#     a = np.array(x.iloc[i].to_list())
#     b = np.array(x.iloc[j].to_list())
#     r = a-b
#     return np.sqrt(np.dot(r,r))

# def heat(x, measure='euclidean'):
#     N = len(x)
#     x_num = x[x.describe().columns]
#     d = distance.pdist(x_num,measure)
#     dist_map = np.zeros((N,N))
#     for i in range(N):
#         for j in range(i):
#             dist_map[i,j] = dist(x_num, i,j)
            
#     return dist_map

###########
# import seaborn as sns
#
#   ...
# dist_map = heat()
# sns.heatmap(dist_map)


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
    st.table(steel_properties_subset.style.background_gradient(cmap='bone'))

    steel_selection = st.sidebar.selectbox("View steels similar to: ", steel_names)
    steel_cluster = get_steel_cluster(steel_data, steel_selection)
    steel_cluster.columns = ['Steel', 'Hardness', 'distance']
    steel_cluster.sort_values(['distance'], ignore_index=True, inplace=True)
    st.sidebar.table(steel_cluster.style.background_gradient(cmap='cividis'))

    # composition_columns = ['C', 'Mn', 'Cr', 'Ni', 'V', 'Mo', 'W', 'Co', 'Si', 'S', 'P']
    # composition_df = steel_properties[composition_columns]
    # #d = distance.pdist(composition_df, 'euclidean')
    # N = 5
    # steel_names_abbr = steel_names_abbr[0:N]
    # dist_map = heat(composition_df.head(N))
    # dist_map_df = pd.DataFrame(data=dist_map, index=steel_names_abbr, columns=steel_names_abbr)
    # sns.heatmap(dist_map_df)

if __name__ == '__main__':

    data_file_path = "steels_data_clusters15.psv"

    run_app(data_file_path)