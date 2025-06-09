import streamlit as st
import pickle
import pandas as pd
import os

# IPL logo in sidebar
st.sidebar.image(
    "https://png.pngtree.com/png-clipart/20190516/original/pngtree-ipl-indian-premier-league-logo-png-png-image_4103353.jpg",
    use_column_width=True
)

# Team and city lists
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load model pipeline with error handling
model_path = 'pipe.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the app directory.")
    st.stop()
else:
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)

st.title('🏏 IPL Win Probability Predictor')

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    # Remove selected batting team from bowling options
    available_bowling_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('Select the bowling team', sorted(available_bowling_teams))

# City selection
selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target Score', min_value=1, max_value=300, step=1, help="Total runs the batting team needs to chase.")

# Match state inputs
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, max_value=target, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Predict button
if st.button('Predict Probability'):
    # Input validation
    if overs == 0:
        st.warning("Overs completed cannot be zero.")
    elif score > target:
        st.warning("Score cannot be greater than the target.")
    elif wickets > 10:
        st.warning("Wickets cannot exceed 10.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # DataFrame for model input
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        try:
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            st.success(f"**{batting_team} Win Probability:** {round(win*100)}%")
            st.info(f"**{bowling_team} Win Probability:** {round(loss*100)}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
