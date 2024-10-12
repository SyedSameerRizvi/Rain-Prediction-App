import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import missingno as msno

# Load the trained model
with open('final.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load dataset
data = pd.read_csv('updated_usa_rain.csv')

# Load the original dataset
df = pd.read_csv('usa_rain.csv')


def feature_engineering(input_data):
    input_data['Temp_Humidity_Interaction'] = input_data['Temperature'] * input_data['Humidity']
    input_data['Wind_Cloud_Ratio'] = input_data['Wind Speed'] / (input_data['Cloud Cover'] + 1e-6)  # Avoid division by zero
    return input_data

# Sidebar: User Inputs
st.sidebar.header("Rain Prediction Input Features")

location = st.sidebar.selectbox('Location', data['Location'].unique())
temperature = st.sidebar.slider('Temperature (°C)', min_value=0, max_value=50, value=25)
humidity = st.sidebar.slider('Humidity (%)', min_value=0, max_value=100, value=50)
wind_speed = st.sidebar.slider('Wind Speed (km/h)', min_value=0, max_value=100, value=10)
precipitation = st.sidebar.slider('Precipitation (mm)', min_value=0.0, max_value=100.0, value=5.0)
cloud_cover = st.sidebar.slider('Cloud Cover (%)', min_value=0, max_value=100, value=50)
pressure = st.sidebar.slider('Pressure (hPa)', min_value=900, max_value=1100, value=1013)

# Get the current date for Year, Month, Day
data['Date'] = pd.to_datetime(data['Date'])
today = datetime.today()  # Use the correct method to get the current date and time
year = today.year
month = today.month
day = today.day

# Create a dataframe for the user input
user_input = pd.DataFrame({
    'Date': [today],  # Date can be today's date or a selectable date
    'Location': [location],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Wind Speed': [wind_speed],
    'Precipitation': [precipitation],
    'Cloud Cover': [cloud_cover],
    'Pressure': [pressure],
    'Year': [year],
    'Month': [month],
    'Day': [day]
})

# Apply feature engineering to add interaction terms
user_input = feature_engineering(user_input)

# Ensure the order of columns matches the training data
columns_needed = ['Date', 'Location', 'Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 
                  'Cloud Cover', 'Pressure', 'Temp_Humidity_Interaction', 'Wind_Cloud_Ratio', 
                  'Year', 'Month', 'Day']

# Reorder the user_input DataFrame to ensure the correct column order
user_input = user_input[columns_needed]

# Function for prediction
def predict_rain(input_data):
    # Drop 'Date' as raw date isn't used, only derived features (Year, Month, Day)
    input_data = input_data.drop(columns=['Date'])
    prediction = model.predict(input_data)
    return 'Rain' if prediction == 1 else 'No Rain'


# Main panel
st.title('Rain Prediction Application')

# Predict rain based on user input
if st.sidebar.button('Predict'):
    result = predict_rain(user_input)
    st.write(f"Prediction: {result}")


# Display Visualizations from the Notebook
st.subheader('Visualizations')


# Visualization 1: Target Class Distribution
st.write("### Target Class Distribution")
target_counts = data["Rain Tomorrow"].value_counts().sort_index()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=['Salmon', 'Blue'])
ax1.set_title('Target Class Distribution (Pie Chart)')
sns.countplot(x="Rain Tomorrow", data=df, ax=ax2, order=target_counts.index, palette=['Salmon', 'Blue'])
ax2.set_title('Target Class Distribution (Count Plot)')
ax2.set_xlabel('Target Class')
ax2.set_ylabel('Count')
plt.tight_layout()
st.pyplot(fig)

# Visualization 2: Location Distribution
st.write("### Location Distribution")
location_counts = df["Location"].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=location_counts.values, y=location_counts.index, palette="icefire", ax=ax)
ax.set_title("Location Distribution")
ax.set_xlabel("Counts")
ax.set_ylabel("Location")
plt.tight_layout()
st.pyplot(fig)

# Visualization 3: Rain Possibility by Location
st.write("### Rain Possibility Across Locations")
rain_by_location = df.groupby("Location")["Rain Tomorrow"].sum()
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(x=rain_by_location.index, y=rain_by_location.values, palette="viridis", ax=ax)
ax.set_title("Rain Possibility varies across Different Locations", fontsize=14)
ax.set_xlabel("Location")
ax.set_xticklabels(rain_by_location.index, rotation=45)
ax.set_ylabel("Counts")
plt.tight_layout()
st.pyplot(fig)
