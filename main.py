import streamlit as st
from lida import Manager, TextGenerationConfig , llm  
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
import openai
from PIL import Image
from io import BytesIO
import base64
import warnings
import os
import time
import numpy as np
import seaborn as sns
from plotly import graph_objects as go

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Visualytics", page_icon=":sparkles:",layout="wide")
st.sidebar.image("./assets/video.gif")

st.title(":sparkles: Visualytics")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    
    return Image.open(BytesIO(byte_data))

library = "seaborn"
lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o-mini", use_cache=True)
menu = st.sidebar.selectbox("Menu", ["Dashboard","Analytics", "Code", "Graph","Custom"])
file_uploader = st.file_uploader(":file_folder: Upload Your File", type=(["csv","xlsx","xls","tsv"]) )

if file_uploader is not None:
    st.write("file_uploader.type",file_uploader.type) 
    if file_uploader.type == "text/csv":
        df = pd.read_csv(file_uploader)
    elif file_uploader.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file_uploader)
    elif file_uploader.type == "application/octet-stream":
        df = pd.read_csv(file_uploader, sep='\t')
    elif file_uploader.name.endswith('.xlsx'):  
        df = pd.read_excel(file_uploader)
    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
            
if menu == "Dashboard":
    if file_uploader is not None:
        st.subheader("Summarization of your Data")
        st.write("") 
        st.write("")
        st.info("Dataset Preview")
        
        st.dataframe(df.head(50), use_container_width=True)

        st.write("") 
        st.write("")

        st.info("Dataset Shape")
        st.write('Number of Rows:', df.shape[0])
        st.write('Number of Columns:', df.shape[1])

        st.write("") 
        st.write("")
        st.info("Column Names and Data Types:")
        st.dataframe(df.dtypes, use_container_width=True)

        st.write("") 
        st.write("")

        if df.isnull().sum().sum() > 0:
            st.info("Null Values")
            st.dataframe(df.isnull().sum().sort_values(ascending=False),width=500)
            st.write("") 
            st.write("")
        else:
            st.info("No Null Values")
            st.write("")
        
        
        st.info("Summary Statistics")
        st.write(df.describe())
        
        st.write("") 
        st.write("")
        st.info("JSON View")
        st.write(summary)

elif menu == "Analytics":
    st.subheader("Analysis of the Data")
    if file_uploader is not None:
          
        st.write("") 
        st.write("")
        summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)

        st.info("Analysis of the Dataset")
        goals = lida.goals(summary, n=5, textgen_config=textgen_config)
        
       
        def show_Analytics():
            count =0
            for goal in goals:
                st.write("Q: ",goal.question)
                st.write("Id: ",goal.visualization)
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                charts = lida.visualize(summary=summary, goal=goal.visualization, textgen_config=textgen_config, library=library)  
                
                explanation = lida.explain(code=charts[0].code, library=library, textgen_config=textgen_config)
                chart_exp = None
                for section in explanation[0]:
                    if section["section"] == "accessibility":
                        chart_exp = section["explanation"]
                st.write("Explanation: ",chart_exp)

                img_base64_string = charts[0].raster
                img = base64_to_image(img_base64_string)
                st.image(img)   
                
                count=count+3
                st.info("-------------------------------------------------------------------------------------------x-------------------------------------------------------------------------------------------")
                if count % 3==0:
                    time.sleep(3)
            
        while True:
            show_Analytics()

elif menu == "Code":
    st.subheader("Query your Data to Generate Code")
    if file_uploader is not None:
        
        text_area = st.text_area("Query your Data to Generate Code", height=200)
        code=0
        if st.button("Generate Code"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                user_query = text_area
                try:
                    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
                    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
                    explanation = lida.explain(code=charts[0].code, library=library, textgen_config=textgen_config)
                    code_exp = None
                    for section in explanation[0]:
                        if section["section"] == "visualization":
                            code_exp = section["explanation"]
                    st.write("Explanation: ",code_exp)
                    code = charts[0].code
                    st.code(code)
                except Exception as e:
                    raise ValueError("The request is Not properly specified or Not genuine: {}".format(str(e)))

elif menu == "Graph":
    st.subheader("Query your Data to Generate Graph")

    if file_uploader is not None:
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                try:
                    image_base64 = charts[0].raster
                    img = base64_to_image(image_base64)
                    st.image(img)
                except Exception as e:
                    raise ValueError("The request is not properly specified or Not genuine: {}".format(str(e)))            
elif menu == "Custom":

    vst = st.selectbox("Select Visualization technique",["Bar","Pie","Pairplot","Heatmap","Linechart", "Treemap"])
    
    if vst =="Bar":
        #bar chart 
        xaxis_options = ['<select>'] + df.columns.tolist()
        yaxis_options = ['<select>'] + df.columns.tolist()

        
        xaxis = st.selectbox("Select x axis column (usually categorical)", xaxis_options)
        yaxis = st.selectbox("Select y axis column (usually numerical)",yaxis_options)

        if xaxis != '<select>' and yaxis != '<select>':
            selected_columns_df = df[[xaxis, yaxis]]

            st.subheader("Bar Chart")
            fig = px.bar(df, x = xaxis, y = yaxis, text =yaxis,template = library)
            st.plotly_chart(fig,use_container_width=True, height = 200)
            csv = selected_columns_df.to_csv(index = False).encode('utf-8')

            st.write(selected_columns_df.T.style.background_gradient(cmap="Blues"))

            st.download_button("Download Data", data = csv, file_name = "Barchart.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')
    
    elif vst=="Pie":
        
        values_options = ['<select>'] + df.columns.tolist()
        labels_options = ['<select>'] + df.columns.tolist()

        values = st.selectbox("Select y axis column (usually numerical)", values_options)
        labels = st.selectbox("Select x axis column (usually categorical)", labels_options)

        if values != '<select>' and labels != '<select>':
            selected_columns_df = df[[values, labels]]

            st.subheader("Pie Chart")
            fig = px.pie(df, names=labels, values=values, template=library)
            st.plotly_chart(fig, use_container_width=True, height=200)
            csv = selected_columns_df.to_csv(index=False).encode('utf-8')

            st.write(selected_columns_df.T.style.background_gradient(cmap="icefire"))

            st.download_button("Download Data", data=csv, file_name="Piechart.csv", mime="text/csv",
                            help='Click here to download the data as a CSV file')
    
    elif vst=="Pairplot":
        st.subheader('Pairplot')

        labels_options = ['<select>'] + df.columns.tolist()
        hue_column = st.selectbox('Select a column to be used as hue', labels_options)

        kind_options = ['scatter', 'reg', 'kde', 'hex']
        kind = st.selectbox('Select the kind of plot', kind_options)

        if hue_column != '<select>':
            for col1 in df.columns[:-1]:
                for col2 in df.columns[:-1]:
                    if col1 != col2:
                        st.write("") 
                        st.subheader(f"Pairplot for {col1} vs {col2}")
                        st.pyplot(sns.pairplot(df, hue=hue_column, x_vars=[col1], y_vars=[col2], kind=kind, palette="dark",height=2,aspect=1.2))
                        st.write("") 
                    
                    
    elif vst=="Heatmap":
        st.subheader('Heatmap')

    # Select the columns which are numeric and then create a correlation matrix
        numeric_columns = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numeric_columns].corr()

        # Convert the seaborn heatmap plot to a Plotly figure
        heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                                x=corr_matrix.columns,
                                                y=corr_matrix.columns,
                                                colorscale='Viridis'))
        st.plotly_chart(heatmap_fig)

        st.write(corr_matrix.T.style.background_gradient(cmap="plasma"))

        csv = corr_matrix.to_csv(index=False).encode("utf-8")
        st.download_button('Download Data', data=csv, file_name="HeatmapData.csv", mime='text/csv')

    elif vst == "Linechart":
        # Line chart
        xaxis_options = ['<select>'] + df.columns.tolist()
        yaxis_options = ['<select>'] + df.columns.tolist()

        xaxis = st.selectbox("Select x-axis column (usually categorical)", xaxis_options)
        yaxis = st.selectbox("Select y-axis column (usually numerical)", yaxis_options)

        if xaxis != '<select>' and yaxis != '<select>':
            linechart = pd.DataFrame(df.groupby(df[xaxis])[yaxis].sum()).reset_index()

            fig2 = px.line(linechart, x=xaxis, y=yaxis, height=500, width=1000, template="gridon")
            st.plotly_chart(fig2, use_container_width=True)

            st.write(linechart.T.style.background_gradient(cmap="twilight"))

            csv = linechart.to_csv(index=False).encode("utf-8")
            st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')

    elif vst=="Treemap":
        xaxis_options = ['<select>'] + df.columns.tolist()
        yaxis_options = ['<select>'] + df.columns.tolist()

        labels = st.selectbox("Select x-axis column (usually categorical)", xaxis_options)
        values = st.selectbox("Select y-axis column (usually numerical)", yaxis_options)
        
        if values != '<select>' and labels != '<select>':
            selected_columns_df = df[[labels,values]]

            st.subheader("Treemap Chart")
            fig = px.treemap(df, path=[labels], values=values)
            st.plotly_chart(fig, use_container_width=True)

            st.write(selected_columns_df.T.style.background_gradient(cmap="hsv"))

            csv = selected_columns_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", data=csv, file_name="Treemap.csv", mime="text/csv",
                            help='Click here to download the data as a CSV file')
            
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
custom_css = """
<style>
div[data-testid="stStreamlitHeader"] div[role="menu"] > div:last-child {
    display: none !important;
}
.st-AboutButton {
    display: none !important;
}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
