# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import july
import base64
from datetime import datetime
import plotly.express as px

# Set app properties
st.set_page_config(
    page_title="Hotel Booking Optimizer",
    page_icon="🏨",  
    layout="wide"  
)

# Cache the model loading function to optimize performance
@st.cache_data()
def load_model():
    filename = "model.sav"  # Model file name
    loaded_model = pickle.load(open(filename, "rb"))  # Load the model using pickle
    return loaded_model

# Function to generate a download link for a DataFrame in CSV format
def get_binary_file_downloader_html(dataframe):
    csv = dataframe.to_csv(index=False)  # Convert DataFrame to CSV
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV to base64
    href = f'''
        <a href="data:file/csv;base64,{b64}" download="hotel_data_with_no_show_prediction.csv" 
        style="
            display:inline-block;
            padding:10px 20px;
            font-size:16px;
            font-weight:bold;
            color:white;
            background-color:#4CAF50;
            border:none;
            border-radius:5px;
            text-decoration:none;
            text-align:center;
        ">
        📥 Download CSV File</a>
    '''
    return href

# App header
st.title("Hotel Booking Optimizer")
st.markdown("🏨🛏️💸 Optimize your hotel room bookings and maximize revenue through strategic overbooking! 🏨🛏️💸")

# Upload and process booking data
st.header("Step 1: Upload Booking Data")
uploaded_file = st.file_uploader("Choose a booking data file (in CSV format)", type="csv")

st.markdown("<br>", unsafe_allow_html=True)
st.header("Step 2: Enter Hotel Details")
max_rooms = st.number_input("Enter total number of rooms in hotel", min_value=0, step=1, value=100)

# Display some empty space
st.markdown("<br>", unsafe_allow_html=True)

# Function to render and process the uploaded data
def render_data(uploaded_file, max_rooms):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)  # Read CSV file into DataFrame
        data['date'] = pd.to_datetime(data['full_date'], format='%d.%m.%Y')  # Convert date column
        data.drop(columns=['full_date'], inplace=True)  # Drop the original date column

        model = load_model()  # Load the prediction model

        # Define expected categories for categorical variables
        expected_categories = {
            'market_segment_type': ['Complementary', 'Online', 'Offline', "Corporate", "Aviation"],
            'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
            'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
        }
        
        # One-hot encode categorical variables
        data = pd.get_dummies(data, columns=['market_segment_type', 'type_of_meal_plan', 'room_type_reserved'], drop_first=False)
        
        # Ensure all expected categories are present in the data
        for column, categories in expected_categories.items():
            for category in categories:
                expected_col = f'{column}_{category}'
                if expected_col not in data.columns:
                    data[expected_col] = 0

        date_series = data['date']  # Store the date series
        model_data = data.drop(columns=['date'], errors='ignore')  # Drop date column for model input
        model_columns = model.feature_names_in_  # Get model feature names
        model_data = model_data.reindex(columns=model_columns, fill_value=0)  # Reorder columns to match model

        predictions = model.predict(model_data)  # Predict no-shows
        data['is_no_show'] = predictions  # Add predictions to data
        data['date'] = date_series  # Restore date column

        # Summarize data by date
        summary_data = data.groupby('date').agg({'is_no_show': 'sum', 'avg_price_per_room': 'mean'}).reset_index()
        summary_data['total_bookings'] = data.groupby('date').size().values  # Count total bookings per date

        # Calculate expected shows and overbooked rooms
        summary_data['expected_shows'] = summary_data['total_bookings'] - summary_data['is_no_show']
        summary_data['overbooked_rooms'] = summary_data['is_no_show']
        summary_data['overbooked_rooms'] = summary_data['overbooked_rooms'].clip(lower=0)  # Clip negative values to 0

        # Calculate additional possible revenue considering the max number of rooms
        summary_data['additional_possible_revenue'] = summary_data.apply(
            lambda row: max(0, min(row['overbooked_rooms'], max_rooms - row['expected_shows']) * row['avg_price_per_room']), axis=1
        )

        # Step 3: Select Analysis Period and Display Data
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Step 3: View Analysis Results")
        period = st.selectbox("Choose the period for the analysis:", ["Daily", "Weekly", "Monthly", "Yearly"])
        
        # Resample data based on selected period
        if period == "Daily":
            display_data = summary_data
        elif period == "Weekly":
            display_data = summary_data.set_index('date').resample('W').sum().reset_index()
        elif period == "Monthly":
            display_data = summary_data.set_index('date').resample('M').sum().reset_index()
        elif period == "Yearly":
            display_data = summary_data.set_index('date').resample('Y').sum().reset_index()

        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%d-%m-%Y')  # Format date
        display_columns = ['date', 'expected_shows', 'is_no_show', 'additional_possible_revenue']

        st.dataframe(display_data[display_columns].rename(columns={
            'date': 'Date', 'expected_shows': 'Predicted Shows', 'is_no_show': 'Predicted No-Shows', 'additional_possible_revenue': 'Additional Possible Revenue'
        }))

        # Generate the heatmap for no-shows
        if not summary_data.empty:
            summary_data['date'] = pd.to_datetime(summary_data['date'], format='%d-%m-%Y')
            start_date = summary_data['date'].min()
            end_date = summary_data['date'].max()

            if pd.notnull(start_date) and pd.notnull(end_date):
                dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()

                if dates:
                    values = summary_data.set_index('date').reindex(dates, fill_value=0)['is_no_show'].tolist()
                    fig, ax = plt.subplots(figsize=(20, 15)) 
                    july.heatmap(
                        dates,
                        values,
                        ax=ax,
                        title="Daily Predicted No-Shows",
                        cmap="viridis",
                        month_grid=True,
                        colorbar=True,
                        dpi=100
                    )
                    st.pyplot(fig)
            else:
                st.error("No dates available to generate heatmap.")
        else:
            st.error("No data available for analysis.")

        # Generate the bar chart for potential revenue
        st.markdown("<br>", unsafe_allow_html=True)
        if not summary_data.empty:
            summary_data['date'] = pd.to_datetime(summary_data['date'], format='%d-%m-%Y')
            fig = px.bar(summary_data, x='date', y='additional_possible_revenue',
                        labels={'date': 'Date', 'additional_possible_revenue': 'Actual Revenue'},
                        title="Potential Additional Revenue from Bookings")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available to generate the potential revenue chart.")

        # Download CSV button
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Step 4: Download Processed Data")
        csv_download_link = get_binary_file_downloader_html(data)
        st.markdown(csv_download_link, unsafe_allow_html=True)

# If a file is uploaded, process the data
if uploaded_file is not None:
    render_data(uploaded_file, max_rooms)

# Section for manually inputting booking variables and predicting show/no-show
st.markdown("<br>", unsafe_allow_html=True)
st.header("Predict No-Show for a Single Booking")

# Load the model
model = load_model()

# Define input fields for booking variables
col1, col2, col3, col4 = st.columns(4)

with col1:
    no_of_adults = st.number_input("Number of Adults", min_value=1, step=1)
    no_of_children = st.number_input("Number of Children", min_value=0, step=1)
    room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    repeated_guest = st.selectbox("Repeated Guest", ["No", "Yes"])

with col2:
    no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, step=1)
    no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, step=1)
    type_of_meal_plan = st.selectbox("Type of Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox("Required Car Parking Space", ["No", "Yes"])

with col3:
    no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, step=1)
    no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Cancelled", min_value=0, step=1)
    market_segment_type = st.selectbox("Market Segment Type", ['Complementary', 'Online', 'Offline', 'Corporate', 'Aviation'])
    no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, step=1)

with col4:
    lead_time = st.number_input("Lead Time (Days)", min_value=0, step=1)
    arrival_date = st.date_input("Arrival Date")  # Single field to input the entire date

# Slider for average room price
avg_price_per_room = st.slider("Average Room Price", min_value=0.0, max_value=400.0, step=0.01, value=200.0)

# Predict show/no-show for the input booking
if st.button("Predict Show/No-Show"):
    # Prepare input data for prediction
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': 1 if required_car_parking_space == "Yes" else 0,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_date.year,
        'arrival_month': arrival_date.month,
        'arrival_day': arrival_date.day,
        'market_segment_type': market_segment_type,
        'repeated_guest': 1 if repeated_guest == "Yes" else 0,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }

    # Convert input data to DataFrame for consistency with model input
    input_df = pd.DataFrame([input_data])

    # Handle categorical features to match training
    input_df = pd.get_dummies(input_df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=False)
    expected_categories = {
        'type_of_meal_plan': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
        'room_type_reserved': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
        'market_segment_type': ['Complementary', 'Online', 'Offline', 'Corporate', 'Aviation']
    }
    for column, categories in expected_categories.items():
        for category in categories:
            expected_col = f'{column}_{category}'
            if expected_col not in input_df.columns:
                input_df[expected_col] = 0

    # Reorder columns to match model input
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict show/no-show
    prediction = model.predict(input_df)

    # Display prediction result
    if prediction == 1:
        st.write("⛔ The booking is predicted to be a no-show. ⛔")
    else:
        st.write("✅ The booking is predicted to be a show. ✅")
