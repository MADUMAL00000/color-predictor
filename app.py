import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Contrarian Roulette Predictor", layout="centered")

st.title("🎯 Contrarian Least-Backed Roulette Predictor")
st.markdown("Rules: The color with fewer bets/occurrences wins, and the heavily backed color loses.")

# European Roulette Official Color Mappings
RED_SET = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_SET = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

def get_color(num):
    if num == 0:
        return "Green"
    elif num in RED_SET:
        return "Red"
    elif num in BLACK_SET:
        return "Black"
    return "Unknown"

COLOR_MAP = {"Red": 0, "Black": 1, "Green": 2}

if 'history_numbers' not in st.session_state:
    st.session_state.history_numbers = []

st.sidebar.subheader("Data Input & Simulation")
input_text = st.sidebar.text_area(
    "Enter Roulette Numbers (up to 500, comma-separated)",
    "34, 16, 26, 28, 23, 24, 6, 9, 18, 24, 4, 29, 14, 35, 35, 36, 16, 35",
    height=200
)

if st.sidebar.button("Run Contrarian Engine"):
    try:
        raw_nums = [int(n.strip()) for n in input_text.split(",") if n.strip().isdigit()]
        raw_nums = [n for n in raw_nums if 0 <= n <= 36]
        
        if len(raw_nums) < 5:
            st.error("කරුණාකර අවම වශයෙන් වලංගු අංක 5ක්වත් ඇතුළත් කරන්න.")
        else:
            st.session_state.history_numbers = raw_nums[-500:]
            st.success(f"ದත්ත සාර්ථකව ගබඩා කරන ලදී! (මුළු අංක: {len(st.session_state.history_numbers)})")
            
    except Exception as e:
        st.error(f"දෝෂයක් සිදු විය: {e}")

if len(st.session_state.history_numbers) >= 5:
    history = st.session_state.history_numbers
    colors = [get_color(n) for n in history]
    
    # Calculate frequencies (simulating betting volume: higher count = heavily backed, lower count = least backed)
    red_count = colors.count("Red")
    black_count = colors.count("Black")
    green_count = colors.count("Green")
    total_count = len(colors)
    
    # Contrarian Logic: Invert the frequencies so that the least frequent gets the highest winning probability
    # Adding 1 to avoid division by zero
    inv_red = 1 / (red_count + 1)
    inv_black = 1 / (black_count + 1)
    inv_green = 1 / (green_count + 1)
    
    inv_total = inv_red + inv_black + inv_green
    contrarian_probs = np.array([inv_red / inv_total, inv_black / inv_total, inv_green / inv_total])
    
    st.markdown("### 📊 Next-Round Contrarian Prediction (අඩුවෙන් ඔට්ටු තබා ඇති පැත්තට වැඩි ඉඩක්)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Red Win Prob.", f"{contrarian_probs[0]*100:.2f}%")
    col2.metric("⚫ Black Win Prob.", f"{contrarian_probs[1]*100:.2f}%")
    col3.metric("🟢 Green Win Prob.", f"{contrarian_probs[2]*100:.2f}%")
    
    prob_df = pd.DataFrame({
        "Color": ["Red (රතු)", "Black (කළු)", "Green (කොළ)"],
        "Recent Occurrences": [red_count, black_count, green_count],
        "Contrarian Win Probability (%)": [f"{p*100:.2f}%" for p in contrarian_probs]
    })
    st.table(prob_df)
    
    st.markdown("### 🧠 Engine Logic Insights")
    st.write("මෙම එන්ජිම මඟින් වැඩියෙන් මතුවී ඇති (ಹೆതുවෙන් ඔට්ටු තබා ඇති යැයි සැලකෙන) වර්ණයට අඩු සම්භාවිතාවකුත්, අඩුවෙන් මතුවී ඇති (අඩුවෙන් ඔට්ටු තබා ඇති) වර්ණයට වැඩි ජයග්‍රහණ සම්භාවිතාවකුත් ලබා දී ඇත.")
    
    if st.button("Clear Memory / දත්ත මකන්න"):
        st.session_state.history_numbers = []
        st.rerun()
else:
    st.info("කරුණාකර වම්පසින් අංක ඇතුළත් කර 'Run Contrarian Engine' බොත්තම ක්ලික් කරන්න.")
