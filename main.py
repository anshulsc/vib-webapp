import streamlit as st
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sb
import plotly.offline as py
from Plotvib import make_segments , convert_df , plot_fft
from Plotvib import plot_data , filter_freq, plot_box
from Plotvib import extract_features, plot_features
import scipy.fft as sp


# Initialize the app
# Configure page settings

st. set_page_config(layout="wide")
box = st.container()




# helper functions

def read_mat(file):
    data = scipy.io.loadmat(file)
    return data

def read_csv(file):
    data = pd.read_csv(file)
    return data

def process_file(file,col):

    file_extension = file.name.split(".")[-1]

    if file_extension == "mat":
 
        data = read_mat(file)
        keys = list(data.keys())
        if len(keys) > 0:
            # col.write("File Name:", file.name)
            selected_key = col.selectbox("Select a key", keys)
            selected_data = np.array(data[selected_key]).reshape(-1)[:120000]

            # df = selected_data.shape
            # col.write("Selected Key:", selected_key)
            # col.write("Shape:", df)
            col.markdown("---")
            return file.name.split(".")[0], selected_key, selected_data
        else:
            col.write("MAT File does not contain any keys.")

    elif file_extension == "csv":

        data = read_csv(file)
        selected_data = np.array(data).reshape(-1)[:120000]
        return file.name.split(".")[0], selected_data
    
    else:
        col.error(f"Invalid file format: {file_extension}. Only MAT files are supported.")


extracted_files = []
extracted_file_names = []


def plot_time(extracted_files, extracted_file_names, index, div):
    i = list(index)
    for k in i:
        a = make_segments(extracted_files[k], 10)
        fig = plot_data(df=a, title=extracted_file_names[0][k])
        div.plotly_chart(fig)
       
        # cont.write("Shape:", extracted_files[k].shape)
        div.markdown("---")

def plot_freq(extracted_files, extracted_file_names, index, filtered_freq, div, freq_range):

    i = list(index)
    frequency = abs(sp.fftfreq(12000,1/12000))
    if filtered_freq:
        for k in i:
            a = make_segments(extracted_files[k], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            c = filter_freq(b)
            fig = plot_fft(x_axis=frequency[:int(freq_range)], y_axis=c, title=extracted_file_names[0][k])
            div.plotly_chart(fig)
     
    else:
        for k in i:
            a = make_segments(extracted_files[k], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            fig = plot_fft(x_axis=frequency[:int(freq_range)], y_axis=b,title=extracted_file_names[0][k])
            div.plotly_chart(fig)

def plot_feat(extracted_files, extracted_file_names, index, filtered_freq,stats_features, div):
    keys = stats_features
    ix = list(index)
    name = [extracted_file_names[0][k] for k in ix]
    frequency = abs(sp.fftfreq(12000,1/12000))
    extracted ={}
    if filtered_freq:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            c = filter_freq(b)
            d = extract_features(c)
            extracted[names] = d
    else:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            d = extract_features(b)
            extracted[names] = d
    if len(keys)>0:
        div.plotly_chart( plot_features(extracted, keys=keys))
    else:
        div.plotly_chart(plot_features(extracted))

def box_plot(extracted_files, extracted_file_names, index, filtered_freq,stats_features,div):
    keys = stats_features.lower()
    ix = list(index)
    name = [extracted_file_names[0][k] for k in ix]
    frequency = abs(sp.fftfreq(12000,1/12000))
    extracted ={}
    if filtered_freq:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            c = filter_freq(b)
            d = extract_features(c)
            extracted[names] = d
    else:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], 10)
            a = a - np.mean(a, axis = 1).reshape(-1,1)
            b = convert_df(a)
            d = extract_features(b)
            extracted[names] = d
    if keys:
        div.plotly_chart(
            plot_box(extracted, value=keys)
            )
    else:
        div.plotly_chart(
            plot_box(extracted, value='mean')
            )


#############################-----Enigne Starts Here-----#####################################

def main():

    col_l, col_r = box.columns([0.4,0.6],gap='large')
    col_l.title("File Upload")
  

    uploaded_files = col_l.file_uploader("Upload MAT or CSV files", accept_multiple_files=True)
    col= col_l.columns(4) 
    li = []
    r = []


    if uploaded_files:
        all_files = [file.name for file in uploaded_files]


        def check_file(files, str):
            for file in files:
                if not file.endswith(str):
                    return False
            return True
            

        if check_file(all_files, ".mat"):
            for file , col1 in zip(uploaded_files, col):
                x, z, y = process_file(file,col1)
                li.append(x)
                r.append(z)
                extracted_files.append(y)
    
        elif check_file(all_files, ".csv"):
            for file , col1 in zip(uploaded_files, col):
                x, y = process_file(file, col1)
                li.append(x)
                extracted_files.append(y)
        else:
            st.error("Please upload only CSV files.")


        extracted_file_names.append(li)
        extracted_file_names.append(r)

    

    if uploaded_files:
        col3,col4 = col_l.columns(2)
        domain = col3.selectbox("What to plot", ['',"Time Domain", "Frequency Domain",'Features','Box Plot'])


        if domain == "Time Domain":
            selected_file_name = col4.multiselect("Select file(s)", extracted_file_names[0])
            
            if selected_file_name:
            
                selected_file_index  = [extracted_file_names[0].index(i) for i in selected_file_name]
                plot_time(extracted_files, extracted_file_names, selected_file_index, col_r)
            else:
                st.write("No file selected")

        if domain == "Frequency Domain":
            selected_file_name = col4.multiselect("Select file(s)", extracted_file_names[0])
            filtered_freq = col4.checkbox("Filter Frequency")
            freq_range = col3.slider("Frequency Range", 200, 6000, 6000)
            
            if selected_file_name:
            
                selected_file_index  = [extracted_file_names[0].index(i) for i in selected_file_name]
                plot_freq(extracted_files, extracted_file_names, selected_file_index, filtered_freq, col_r,freq_range)
            else:
                col_l.write("No file selected")


        if domain == "Features":
            selected_file_name = col4.multiselect("Select file(s)", extracted_file_names[0])
            #multiselct stats features but only 4 can be selected at a time
            stats_features = col4.multiselect("Select stats features", ['Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis','shape_factor','impulse_factor'], default=[], key="stats_features")
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time
            if len(stats_features) > 4:
                st.error("Please select only 4 stats features.")
            
            else:
                if selected_file_name:
                
                    selected_file_index  = [extracted_file_names[0].index(i) for i in selected_file_name]
                    plot_feat(extracted_files, extracted_file_names, selected_file_index, filtered_freq,stats_features,col_r)
                else:
                    st.write("No file selected")

        if domain == "Box Plot":
            selected_file_name = col4.multiselect("Select file(s)", extracted_file_names[0])
            #multiselct stats features but only 4 can be selected at a time
            stats_features = col4.selectbox("Select stats features", [''] + ['Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis','shape_factor','impulse_factor'])
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time
        
            

            if selected_file_name:
                    selected_file_index  = [extracted_file_names[0].index(i) for i in selected_file_name]
                    box_plot(extracted_files, extracted_file_names, selected_file_index, filtered_freq,stats_features,col_r)
            else:
                    st.write("No file selected")

        with col_l:
            if col_l.button("Show Files"):
                st.write("Extracted Files:", extracted_files)
                st.write("Extracted File Names:", extracted_file_names)
     
        # Perform actions with extracted_files and extracted_file_names
        # Example: Save files or perform further processing
        
        # col_l.markdown(f"Extracted Files: {extracted_files}")
        # col_l.markdown("Extracted File Names:", extracted_file_names)

     

if __name__ == "__main__":
    main()
