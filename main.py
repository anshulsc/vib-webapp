import streamlit as st
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sb
import plotly.offline as py
from Plotvib import make_segments, convert_df, plot_fft
from Plotvib import plot_data, filter_freq, plot_box
from Plotvib import extract_features, plot_features, envelope_plot
import scipy.fft as sp


# Initialize the app
# Configure page settings

st.set_page_config(layout="wide")
box = st.container()


# helper functions


def read_mat(file):
    data = scipy.io.loadmat(file)
    return data


def read_csv(file):
    data = pd.read_csv(file)
    return data


def process_file(file, col):
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

            return file.name.split(".")[0], selected_key, selected_data
        else:
            col.write("MAT File does not contain any keys.")

    elif file_extension == "csv":
        data = read_csv(file)
        selected_data = np.array(data).reshape(-1)[:120000]
        return file.name.split(".")[0], selected_data

    else:
        col.error(
            f"Invalid file format: {file_extension}. Only MAT files are supported."
        )


extracted_files = []
extracted_file_names = []


def plot_time(
    extracted_files, extracted_file_names, index, div, total_segment, seg_num , env_plot, isolated
):
    fig = None
    i = list(index)
    try:
        for k in i:
            a = make_segments(extracted_files[k], num_segments=total_segment)
            if env_plot:
                fig = envelope_plot(df=a, title=extracted_file_names[0][k], seg_num=seg_num, show_real= (not isolated))
            else:
                fig = plot_data(df=a, title=extracted_file_names[0][k], seg_num=seg_num)
            div.plotly_chart(fig)

            # cont.write("Shape:", extracted_files[k].shape)
            div.markdown("---")
    except:
        st.error("Please select a right key")


def plot_freq(
    extracted_files,
    extracted_file_names,
    index,
    filtered_freq,
    div,
    freq_range,
    total_segments,
    seg_num,
    limit
):
    i = list(index)
    frequency = abs(sp.fftfreq((120000 // total_segments), 1 / 12000))
    steps = int(frequency[1] - frequency[0])
    try:
        if filtered_freq:
            for k in i:
                a = make_segments(extracted_files[k], num_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, num_segments=total_segments)
                c = filter_freq(b, amp=limit)
                fig = plot_fft(
                    x_axis=frequency[: int(freq_range // steps)],
                    y_axis=c,
                    title=extracted_file_names[0][k],
                    seg_num=seg_num,
                )
                div.plotly_chart(fig)

        else:
            for k in i:
                a = make_segments(extracted_files[k], num_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, num_segments=total_segments)
                fig = plot_fft(
                    x_axis=frequency[: int(freq_range // steps)],
                    y_axis=b,
                    title=extracted_file_names[0][k],
                    seg_num=seg_num,
                )
                div.plotly_chart(fig)
    except:
        st.error("Please select a right key")


def plot_feat(
    extracted_files,
    extracted_file_names,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
):
    keys = stats_features
    ix = list(index)
    name = [extracted_file_names[0][k] for k in ix]
    frequency = abs(sp.fftfreq((12000//total_segments), 1 / 12000))
    extracted = {}
    try:
        if filtered_freq:
            for names, i in zip(name, index):
                a = make_segments(extracted_files[i], num_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, num_segments=total_segments)
                c = filter_freq(b,file_name=names)
                d = extract_features(c)
                extracted[names] = d
        else:
            for names, i in zip(name, index):
                a = make_segments(extracted_files[i], num_segments=total_segments)
                a = a - np.mean(a, axis=1).reshape(-1, 1)
                b = convert_df(a, num_segments=total_segments)
                d = extract_features(b)
                extracted[names] = d
    except:
        st.error("Please select a right key")
    
    if len(keys) > 0:
        div.plotly_chart(plot_features(extracted, keys=[f"{keys}"]))
    else:
        div.plotly_chart(plot_features(extracted))


def box_plot(
    extracted_files,
    extracted_file_names,
    index,
    filtered_freq,
    stats_features,
    div,
    total_segments,
):
    keys = stats_features.lower()
    ix = list(index)
    name = [extracted_file_names[0][k] for k in ix]
    frequency = abs(sp.fftfreq(12000, 1 / 12000))
    extracted = {}
    if filtered_freq:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], num_segments=total_segments)
            a = a - np.mean(a, axis=1).reshape(-1, 1)
            b = convert_df(a, num_segments=total_segments)
            c = filter_freq(b,file_name=names)
            d = extract_features(c)
            extracted[names] = d
    else:
        for names, i in zip(name, index):
            a = make_segments(extracted_files[i], 10)
            a = a - np.mean(a, axis=1).reshape(-1, 1)
            b = convert_df(a)
            d = extract_features(b)
            extracted[names] = d
    if keys:
        div.plotly_chart(plot_box(extracted, value=keys))
    else:
        div.plotly_chart(plot_box(extracted, value="mean"))


#############################-----Enigne Starts Here-----#####################################


def main():

    col_l, col_r = box.columns([0.4, 0.6], gap="large")
    col_l.title("File Upload")
    limit = None

    uploaded_files = col_l.file_uploader(
        "Upload MAT or CSV files", accept_multiple_files=True
    )
    exp = col_l.expander("Select Keys", expanded=True)
    cols = []

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
            if len(uploaded_files) <= 4 and len(uploaded_files) > 0:
                cols = exp.columns(len(uploaded_files))

            else:
                i = len(uploaded_files) // 4 + 1
                for _ in range(i):
                    cols.extend(exp.columns(4))

            for file, col1 in zip(uploaded_files, cols):
                x, z, y = process_file(file, col1)
                li.append(x)
                r.append(z)
                extracted_files.append(y)

        elif check_file(all_files, ".csv"):
            for file in  uploaded_files :
                x, y = process_file(file, col_l)
                li.append(x)
                extracted_files.append(y)
        else:
            st.error("Please upload only CSV or Mat files.")

    extracted_file_names.append(li)
    extracted_file_names.append(r)
    
    
    
    if uploaded_files:
        total_segments = col_l.number_input("Total Number of Segments", 1, 200, 10)
        col3, col4 = col_l.columns(2)
        domain = col3.selectbox(
            "What to plot",
            ["", "Time Domain", "Frequency Domain", "Features", "Box Plot"],
        )

        if domain == "Time Domain":
            selected_file_name = col4.selectbox(
                "Select a file", extracted_file_names[0]
            )
            isolated = None
            env_plot = col4.checkbox("Plot Envelope")
           
            seg_num = col3.number_input("Segment Number", 1, total_segments)

            if env_plot:
                 isolated = col4.checkbox("Isolate Envelope")

            if selected_file_name:
                selected_file_index = [
                    extracted_file_names[0].index(selected_file_name)
                ]
                plot_time(
                    extracted_files,
                    extracted_file_names,
                    selected_file_index,
                    col_r,
                    total_segments,
                    seg_num,
                    env_plot,
                    isolated,
                )
            else:
                st.write("No file selected")

        if domain == "Frequency Domain":
            selected_file_name = col4.selectbox(
                "Select a file", extracted_file_names[0]
            )
            frequency = abs(sp.fftfreq((120000 // total_segments), 1 / 12000))
            steps = frequency[1] - frequency[0]
            seg_num = col4.number_input("Segment Number", 1, total_segments)
            filtered_freq = col4.checkbox("Filter Frequency")
            if filtered_freq:
                limit = col4.number_input("Frequency Limit", 0.0,1.0,0.2,step = 0.01)

            freq_range = col3.slider(
                "Frequency Range", 200, 6000, 6000, step=int(steps)
            )

            if selected_file_name:
                selected_file_index = [
                    extracted_file_names[0].index(selected_file_name)
                ]
                plot_freq(
                    extracted_files,
                    extracted_file_names,
                    selected_file_index,
                    filtered_freq,
                    col_r,
                    freq_range,
                    total_segments,
                    seg_num,
                    limit
                )
            else:
                col_l.write("No file selected")

        if domain == "Features":
            selected_file_name = col4.multiselect(
                 "Select file(s)", extracted_file_names[0])
            # multiselct stats features but only 4 can be selected at a time
            stats_features = col4.selectbox(
                "Select stats features",
                [
                    "Mean",
                    "Max",
                    "Variance",
                    "Skewness",
                    "Kurtosis",
                    "shape_factor",
                    "impulse_factor",
                ],
                key="stats_features",
            )
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                selected_file_index = [ extracted_file_names[0].index(i) for i in selected_file_name ]
                
                plot_feat(
                    extracted_files,
                    extracted_file_names,
                    selected_file_index,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                )
            else:
                st.write("No file selected")

        if domain == "Box Plot":
            selected_file_name = col4.multiselect(
                "Select file(s)", extracted_file_names[0]
            )
            # multiselct stats features but only 4 can be selected at a time
            stats_features = col4.selectbox(
                "Select stats features", 
                 [ "Mean", "Max", "Variance", "Skewness", "Kurtosis", "shape_factor","impulse_factor"],)
            
            filtered_freq = col4.checkbox("Filter Frequency")
            # multiselect time features but only 4 can be selected at a time

            if selected_file_name:
                selected_file_index = [
                    extracted_file_names[0].index(i) for i in selected_file_name
                ]
                box_plot(
                    extracted_files,
                    extracted_file_names,
                    selected_file_index,
                    filtered_freq,
                    stats_features,
                    col_r,
                    total_segments,
                )
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
