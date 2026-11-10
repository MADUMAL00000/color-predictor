import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Roulette Advanced Stateful Hybrid Predictor", layout="centered")

st.title("🎯 Roulette Advanced Stateful Hybrid Predictor")
st.markdown("Automatic Number-to-Color Mapping, Stateful Memory, Deep Pattern Learning & Next-Round Probability Engine.")

# Roulette Color Mappings
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

# Initialize Session State for Internal Memory Persistence
if 'history_numbers' not in st.session_state:
    st.session_state.history_numbers = []
if 'transition_matrix' not in st.session_state:
    st.session_state.transition_matrix = np.zeros((3, 3))

st.sidebar.subheader("Data Input & Stateful Memory")
input_text = st.sidebar.text_area(
    "Enter Roulette Numbers (up to 500, comma-separated)",
    "34, 16, 26, 28, 23, 24, 6, 9, 18, 24, 4, 29, 14, 35, 35, 36, 16, 35",
    height=200
)

if st.sidebar.button("Run & Train Prediction Engine"):
    try:
        # Parse numbers and filter valid 0-36 range
        raw_nums = [int(n.strip()) for n in input_text.split(",") if n.strip().isdigit()]
        raw_nums = [n for n in raw_nums if 0 <= n <= 36]
        
        if len(raw_nums) < 5:
            st.error("කරුණාකර අවම වශයෙන් වලංගු අංක 5ක්වත් ඇතුළත් කරන්න.")
        else:
            # Store up to 500 numbers in internal memory
            st.session_state.history_numbers = raw_nums[-500:]
            
            # Map numbers to colors and encode
            colors = [get_color(n) for n in st.session_state.history_numbers]
            encoded = [COLOR_MAP[c] for c in colors if c in COLOR_MAP]
            
            # Train Stateful Markov Chain Transition Matrix (Hidden Pattern Learning)
            transition_matrix = np.zeros((3, 3))
            for i in range(len(encoded) - 1):
                current_state = encoded[i]
                next_state = encoded[i+1]
                transition_matrix[current_state][next_state] += 1
            
            st.session_state.transition_matrix = transition_matrix
            st.success(f"දත්ත සාර්ථකව ගබඩා කර එන්ජිම පුහුණු කරන ලදී! (මුළු අංක: {len(st.session_state.history_numbers)})")
            
    except Exception as e:
        st.error(f"දෝෂයක් සිදු විය: {e}")

# Main Dashboard & Next-Round Prediction Execution
if len(st.session_state.history_numbers) >= 5:
    history = st.session_state.history_numbers
    colors = [get_color(n) for n in history]
    encoded = [COLOR_MAP[c] for c in colors if c in COLOR_MAP]
    
    # Normalize transition matrix stored in memory
    row_sums = st.session_state.transition_matrix.sum(axis=1, keepdims=True)
    markov_probs = np.where(row_sums > 0, st.session_state.transition_matrix / row_sums, 1/3)
    
    last_state = encoded[-1]
    next_markov_prob = markov_probs[last_state]
    
    # Global Frequency Weighting for Deep Pattern Alignment
    counts = [encoded.count(0), encoded.count(1), encoded.count(2)]
    total = sum(counts)
    freq_probs = np.array([c/total for c in counts]) if total > 0 else np.array([1/3, 1/3, 1/3])
    
    # Hybrid Advanced Calculation (Markov 75% + Frequency 25% for optimal pattern capture)
    final_probs = 0.75 * next_markov_prob + 0.25 * freq_probs
    final_probs = final_probs / np.sum(final_probs)
    
    st.markdown("### 📊 Next-Round Prediction (මීළඟ 1 වටය සඳහා සම්භාවිතාව)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Red (රතු)", f"{final_probs[0]*100:.2f}%")
    col2.metric("⚫ Black (කළු)", f"{final_probs[1]*100:.2f}%")
    col3.metric("🟢 Green (කොළ)", f"{final_probs[2]*100:.2f}%")
    
    # Detailed Probability Table
    prob_df = pd.DataFrame({
        "Color": ["Red (රතු)", "Black (කළු)", "Green (කොළ)"],
        "Probability (%)": [f"{p*100:.2f}%" for p in final_probs]
    })
    st.table(prob_df)
    
    # Stateful Memory & Pattern Insights
    st.markdown("### 🧠 Stateful Engine Memory & Pattern Insights")
    st.write(f"මතකයේ ගබඩා කර ඇති මුළු අංක ගණන: **{len(history)} / 500**")
    st.write(f"අන්තිමට වාර්තා වූ අංකය: **{history[-1]}** (ස්වයංක්‍රීය වර්ණය: **{get_color(history[-1])}**)")
    st.write("වෙනස් වන සැඟවුණු රටා (Hidden State Transitions) ස්වයංක්‍රීයව මතකයේ තබා ගනිමින් විශ්ලේෂණය කරනු ලැබේ.")
    
    if st.button("Clear Memory / දත්ත මකන්න"):
        st.session_state.history_numbers = []
        st.session_state.transition_matrix = np.zeros((3, 3))
        st.rerun()
else:
    st.info("කරුණාකර වම්පසින් අංක ඇතුළත් කර 'Run & Train Prediction Engine' බොත්තම ක්ලික් කරන්න.")
