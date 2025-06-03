import streamlit as st
import joblib

st.set_page_config(page_title="Girlfriend Mood Predictor💕", layout="centered")

# 1. Pink theme & styling
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #ffe4e1;  /* light pink */
        color: #4b0082;             /* dark purple text */
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffb6c1;  /* lighter pink */
    }
    /* Main text */
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div {
        color: #4b0082;
    }
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #fff0f5;  /* very light pink */
        color: #4b0082;
    }
    /* Buttons */
    .stButton>button {
        background-color: #ff69b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ff1493;
        color: white;
    }
    /* Label text */
    label {
        color: #4b0082 !important;
        font-weight: 600;
    }
    /* Result text */
    .result-text {
        color: #8B0000;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Warning banner
st.markdown("""
<div style="background-color:#ffccd5; padding:15px; border-radius:10px; margin-bottom:20px;">
    <h3 style="color:#b30059;">🚨 Warning! 🚨</h3>
    <p style="color:#660033; font-size:13px;">
        This mood predictor is a Machine Learning model trained on thoughtfully crafted emotional and behavioral data.
        ⚠️Results may cause sudden urges to apologize, hug, or buy biryani. Use with care — and lots of love! 💖
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 4. App title and description
st.title("💖Girlfriend Mood Predictor")
st.write("Answer a few simple questions to estimate her mood today!")

# 5. Input fields
bf_name = st.text_input("Your name")
gf_name = st.text_input("Your girlfriend's name")

user_input = {
    "sleep_hours": st.slider("How many hours did she sleep?", 3, 10),
    "gave_compliments": st.selectbox("Did you give her compliments today?", ["-- Select --", "yes", "no"]),
    "unresolved_argument": st.selectbox("Do you have any unresolved argument?", ["-- Select --", "none", "mild", "serious"]),
    "expressed_love": st.selectbox("Did you express your love today?", ["-- Select --", "yes", "no"]),
    "self_care": st.selectbox("Did she spend time for self-care?(exercise, journaling etc.)", ["-- Select --", "yes", "no"]),
    "ate_fav_food": st.selectbox("Did she eat something she loves today?", ["-- Select --", "yes", "no"]),
    "menstrual_phase": st.selectbox("Where is she in her menstrual cycle?", ["-- Select --", "unknown", "early", "mid", "PMS"]),
    "out_with_friends_hrs": st.slider("How many hours were you out with friends without her?", 0, 10),
    "took_bath": st.selectbox("Did she take a bath today?", ["-- Select --", "yes", "no"]),
    "work_pressure": st.selectbox("Does she have work pressure today?", ["-- Select --", "no", "minor", "major"]),
}

# 6. Predict button logic
if st.button("Predict Mood"):
    if "-- Select --" in user_input.values():
        st.error("Please complete all selections.")
    else:
        # Mapping categorical answers to numeric values
        mapping = {
            "yes": 1, "no": 0,
            "none": 0, "mild": 1, "serious": 2,
            "unknown": 0, "early": 1, "mid": 2, "PMS": 3,
            "no": 0, "minor": 1, "major": 2
        }

        encoded_input = [
            user_input["sleep_hours"],
            mapping[user_input["gave_compliments"]],
            mapping[user_input["unresolved_argument"]],
            mapping[user_input["expressed_love"]],
            mapping[user_input["self_care"]],
            mapping[user_input["ate_fav_food"]],
            mapping[user_input["menstrual_phase"]],
            user_input["out_with_friends_hrs"],
            mapping[user_input["took_bath"]],
            mapping[user_input["work_pressure"]],
        ]

        # Scale the inputs and predict
        scaled_input = scaler.transform([encoded_input])
        mood_score = model.predict(scaled_input)[0]
        mood_score = round(mood_score, 1)

        # Categorize mood and generate messages
        if mood_score <= 3:
            status = "😡 Angry/Sad"
            message = f"Hey {bf_name}, {gf_name or 'She'} might need a hug and a biryani🍛. Get it fast before her mood hits 0!"
        elif mood_score <= 5:
            status = "😟 Upset"
            message = f"Hmmmm {bf_name}, {gf_name} seems upset. 😕 Maybe go ask her 'what happened?' for the 5th time today😕."
        elif mood_score <= 6.5:
            status = "😐 Neutral"
            message = f"Not bad, not great. 😶 Maybe share a meme to boost {gf_name}’s mood, {bf_name}."
        elif mood_score <= 8:
            status = "😊 Happy"
            message = f"{gf_name} is in a good mood today! 😊 Just don’t mess it up, {bf_name} 😎"
        else:
            status = "🤩 Super Happy"
            message = f"{bf_name}, {gf_name} is SUPER happy today! 🥳. Now’s the time to pitch your 'genius' business idea with {gf_name} . She might just support it today! 😜💡"

        # Display results with styling
        st.markdown(
            f"""
            <div class='result-text'>
                <p>🎯 <strong>Mood Score:</strong> {mood_score} / 10</p>
                <p>📊 <strong>Status:</strong> {status}</p>
                <p>💬 {message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
