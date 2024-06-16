import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from PIL import Image
import datetime

# Load the dataset
file_path = 'stationary_df.csv'
data = pd.read_csv(file_path)
data.drop('Unnamed: 0', axis=1, inplace=True)

# Convert 'date_surveyed' to datetime and set as index
data['date_surveyed'] = pd.to_datetime(data['date_surveyed'], format='%Y-%m-%d')
data.set_index('date_surveyed', inplace=True)

# Rename column for clarity
data.rename(columns={'daily_co2_emmission_ppm_stationary': 'daily_co2_emmission_ppm'}, inplace=True)

# Set up the Streamlit app layout
st.set_page_config(layout="wide")

# Display logo and title
st.sidebar.image('images/uniport_logo.png', width=100)
st.sidebar.title("CO2 Emission Dashboard")

# User input via sidebar for granularity and location
granularity = st.sidebar.radio("Time Granularity", ["Daily", "Weekly", "Monthly"])
selected_location = st.sidebar.selectbox("Select Location", data['Area_Surveyed'].unique())

# Function to resample data based on selected granularity
def resample_data(df, granularity):
    rule = granularity[0].upper()  # Get the first letter (D, W, or M)
    return df.resample(rule).mean() 

# Filter and resample data for the selected location
location_data = data[data['Area_Surveyed'] == selected_location]
location_data = resample_data(location_data, granularity)

# Dynamic date display and latest CO2 emission value
filtered_data = data[data['Area_Surveyed'].isin(selected_locations)]

if not filtered_data.empty:
    numeric_columns = filtered_data.select_dtypes(include=np.number)  
    resampled_data = numeric_columns.groupby('Area_Surveyed').apply(lambda x: resample_data(x, granularity))

    st.sidebar.markdown(f"**Latest reading from {latest_date.strftime('%Y-%m-%d')}:**")
    st.sidebar.markdown(f"**{current_co2:.2f} ppm CO2**", unsafe_allow_html=True)

# Main dashboard layout with location and statistics
col1, col2 = st.columns([2, 1])
with col1:
    st.header(f"CO2 Emission Trends for {selected_location}")
    fig = px.line(location_data, y='daily_co2_emmission_ppm', title='Trend Over Time', labels={'value': 'CO2 Emission (ppm)'})
    st.plotly_chart(fig, use_container_width=True)

    # Display location image
    image_path = f'images/{selected_location}.png'  # assuming the file name is the exact location name
    try:
        location_image = Image.open(image_path)
        st.image(location_image, caption=f'Image of {selected_location}')
    except FileNotFoundError:
        st.error("Image file not found.")

with col2:
    st.subheader("Descriptive Statistics")
    st.write(location_data.describe())
    if len(location_data) > 12:  # Conditional display of forecasting
        ets_model = ETSModel(location_data['daily_co2_emmission_ppm'], error='add', trend='add', seasonal='add', seasonal_periods=12)
        ets_fit = ets_model.fit()
        forecast_values = ets_fit.forecast(steps=30)
        forecast_df = pd.DataFrame({'Forecast': forecast_values, 'Date': forecast_values.index})
        forecast_df.set_index('Date', inplace=True)

        st.subheader('Forecasted CO2 Emission')
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=location_data.index, y=location_data['daily_co2_emmission_ppm'], mode='lines', name='Observed'))
        forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
        forecast_fig.update_layout(title='Forecast', xaxis_title='Date', yaxis_title='CO2 Emission (ppm)')
        st.plotly_chart(forecast_fig, use_container_width=True)

        # AI Insights Placeholder function
        def ai_insights(data):
            insights = "Analysis shows trends, peaks, and predictions aiding in proactive measures."
            return insights

        st.subheader("AI-Generated Insights")
        st.write(ai_insights(location_data))

        # Download data button
        st.download_button(label='Download Forecast Data', data=forecast_df.to_csv(), file_name='forecasted_data.csv', mime='text/csv')

# Additional styling for better visual appeal
st.markdown("""
    <style>
    .css-1aumxhk {
        background-color: #f0f2f6;
        border-color: #f0f2f6;
    }
    .css-1d391kg {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


