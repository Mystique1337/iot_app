import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import requests
from PIL import Image
from io import BytesIO
import base64 

# Load the dataset
file_path = 'stationary_df.csv'
data = pd.read_csv(file_path)
data.drop('Unnamed: 0', axis=1, inplace=True)
# Convert date_surveyed to datetime
data['date_surveyed'] = pd.to_datetime(data['date_surveyed'], format='%Y-%m-%d')
data.rename(columns={'daily_co2_emmission_ppm_stationary': 'daily_co2_emmission_ppm'}, inplace=True)

# Title for Streamlit app
st.title("CO2 Emission Dashboard per Location")

# Sidebar for user input
granularity = st.sidebar.selectbox("Select Time Granularity", ["Daily", "Weekly", "Monthly"])
selected_location = st.sidebar.selectbox("Select Location", data['Area_Surveyed'].unique())

# Function to resample data based on granularity
def resample_data(df, granularity):
    if granularity == "Weekly":
        return df.resample('W').mean()
    elif granularity == "Monthly":
        return df.resample('M').mean()
    return df

# Function to get image for location
def get_image_for_location(location):
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=400&titles={location.replace(' ', '_')}"
    response = requests.get(search_url).json()
    pages = response['query']['pages']
    page = next(iter(pages.values()))
    image_url = page.get('thumbnail', {}).get('source', None)
    if image_url:
        return Image.open(BytesIO(requests.get(image_url).content))
    return None

# Filter data for the selected location
location_data = data[data['Area_Surveyed'] == selected_location].copy()
location_data.set_index('date_surveyed', inplace=True)
location_data = resample_data(location_data, granularity)

# Display image for location
image = get_image_for_location(selected_location)
if image:
    st.image(image, caption=f'{selected_location}')

# Descriptive statistics
st.header(f"Descriptive Statistics for {selected_location}")
st.write(location_data.describe())

# Plot the time series
st.subheader(f'Daily CO2 Emission Over Time ({granularity})')
fig = px.line(location_data, y='daily_co2_emmission', title=f'Daily CO2 Emission Over Time - {selected_location}')
st.plotly_chart(fig)

# SARIMA model for better forecasting
sarima_model = SARIMAX(location_data['daily_co2_emmission_ppm'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
forecast = sarima_fit.get_forecast(steps=30)
forecast_df = forecast.conf_int()
forecast_df['Forecast'] = sarima_fit.predict(start=forecast_df.index[0], end=forecast_df.index[-1])



# Plot forecast
st.subheader('CO2 Emission Forecast')
fig = go.Figure()
fig.add_trace(go.Scatter(x=location_data.index, y=location_data['daily_co2_emmission_ppm'], mode='lines', name='Observed'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
fig.update_layout(title=f'CO2 Emission Forecast - {selected_location}', xaxis_title='Date', yaxis_title='CO2 Emission')
st.plotly_chart(fig)

# Button to download plot as PNG
# Encode Plotly figure to base64 image
buffer = BytesIO()
fig.write_image(buffer, format="png")
img_str = base64.b64encode(buffer.getvalue()).decode()

# Create download link
href = f'<a href="data:image/png;base64,{img_str}" download="forecast_plot.png">Download plot as PNG</a>'
st.markdown(href, unsafe_allow_html=True)


# Display forecast values
st.subheader('Forecast Values')
st.write(forecast_df[['Forecast']])

# Safety evaluation
safety_threshold = 0.0005  # Example threshold
max_emission = location_data['daily_co2_emmission_ppm'].max()
safety_status = "Safe" if max_emission < safety_threshold else "Unsafe"

st.subheader('Safety Evaluation')
st.write(f"The highest recorded CO2 emission for {selected_location} is {max_emission:.6f}. The location is considered {safety_status} based on a threshold of {safety_threshold:.6f}.")

# Conversational AI section
st.header("Conversational AI Insights")

def ai_insights(location_data):
    max_value = location_data['daily_co2_emmission_ppm'].max()
    mean_value = location_data['daily_co2_emmission_ppm'].mean()
    std_value = location_data['daily_co2_emmission_ppm'].std()
    
    insights = f"""
    The maximum CO2 emission recorded is {max_value:.6f}, which indicates the peak level of emissions.
    On average, the CO2 emission is {mean_value:.6f}, showing the general trend of emission levels.
    The standard deviation of {std_value:.6f} suggests the variability in the CO2 emission data.
    """
    if max_value > safety_threshold:
        insights += " The emission levels have exceeded the safety threshold at times, which is concerning."
    else:
        insights += " The emission levels are within the safe range."
    
    return insights

st.write(ai_insights(location_data))
