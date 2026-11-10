import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Roulette Advanced Hybrid Predictor", layout="centered")

st.title("🎯 Roulette Advanced Hybrid Predictor")
st.markdown("Automatic Number-to-Color Mapping, Deep Pattern Learning & Next-Round Probability Engine.")

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

st.sidebar.subheader("Data Input & Configuration")
input_text = st.sidebar.text_area(
    "Enter Roulette Numbers (up to 500, comma-separated)",
    "34, 16, 26, 28, 23, 24, 6, 9, 18, 24, 4, 29, 14, 35, 35, 36, 16, 35",
    height=200
)

if st.sidebar.button("Run Next-Round Prediction Engine"):
    try:
        # Parse numbers and filter valid 0-36 range
        raw_nums = [int(n.strip()) for n in input_text.split(",") if n.strip().isdigit()]
        raw_nums = [n for n in raw_nums if 0 <= n <= 36]
        
        if len(raw_nums) < 5:
            st.error("කරුණාකර අවම වශයෙන් වලංගු අංක 5ක්වත් ඇතුළත් කරන්න.")
        else:
            # Support up to 500 numbers
            raw_nums = raw_nums[-500:]
            
            # Map numbers to colors
            colors = [get_color(n) for n in raw_nums]
            encoded = [COLOR_MAP[c] for c in colors if c in COLOR_MAP]
            
            st.success(f"සාර්ථකව අංක {len(raw_nums)}ක් මතකයේ රඳවා ගනිමින් විශ්ලේෂණය කරන ලදී!")
            
            # Markov Chain Hidden Pattern Transition Matrix
            transition_matrix = np.zeros((3, 3))
            for i in range(len(encoded) - 1):
                current_state = encoded[i]
                next_state = encoded[i+1]
                transition_matrix[current_state][next_state] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            markov_probs = np.where(row_sums > 0, transition_matrix / row_sums, 1/3)
            
            last_state = encoded[-1]
            next_markov_prob = markov_probs[last_state]
            
            # Global Frequency Weighting
            counts = [encoded.count(0), encoded.count(1), encoded.count(2)]
            total = sum(counts)
            freq_probs = np.array([c/total for c in counts]) if total > 0 else np.array([1/3, 1/3, 1/3])
            
            # Hybrid Calculation (Markov 70% + Frequency 30% for advanced pattern detection)
            final_probs = 0.7 * next_markov_prob + 0.3 * freq_probs
            final_probs = final_probs / np.sum(final_probs)
            
            st.markdown("### 📊 Next-Round Color Probability (මීළඟ 1 වටය සඳහා සම්භාවිතාව)")
            
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
            
            # Pattern Summary
            st.markdown("### 🔍 Historical Pattern & Memory Summary")
            st.write(f"පද්ධතිය තුළ ගබඩා කර අධ්‍යයනය කළ මුළු අංක ගණන: **{len(raw_nums)}**")
            st.write(f"අන්තිමට යෙදූ අංකය: **{raw_nums[-1]}** (වර්ණය: **{get_color(raw_nums[-1])}**)")
            
    except Exception as e:
        st.error(f"දෝෂයක් සිදු විය: {e}")

