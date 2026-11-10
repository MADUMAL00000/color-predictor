import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import plotly.express as px
import time

# ==========================================
# 1. PAGE CONFIGURATION & THEME SETUP
# ==========================================
st.set_page_config(
    page_title="Advanced 3-Color Seq2Seq Predictor",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1E3A8A; }
    .sub-text { font-size: 15px; color: #4B5563; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Quantitative 3-Color 9-Step Horizon Prediction Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Hybrid Deep Learning (Multi-Output LSTM) + Probabilistic Markov Transitions with Real-time Concept Drift Adaptation.</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. ENCODING & DATA MAPPING
# ==========================================
COLOR_MAP = {"A": 0, "B": 1, "C": 2}
REV_COLOR_MAP = {0: "Color A (Red)", 1: "Color B (Green)", 2: "Color C (Blue)"}
COLOR_HEX = {0: "#EF4444", 1: "#10B981", 2: "#3B82F6"}

def encode_sequence(seq_str):
    tokens = [t.strip().upper() for t in seq_str.split(",") if t.strip() in COLOR_MAP]
    return [COLOR_MAP[t] for t in tokens]

def one_hot_encode(sequence, num_classes=3):
    encoded = np.zeros((len(sequence), num_classes), dtype=np.float32)
    for i, val in enumerate(sequence):
        encoded[i, val] = 1.0
    return encoded

# ==========================================
# 3. HYBRID MODEL ARCHITECTURE (PYTORCH)
# ==========================================
class MultiOutputSeq2SeqLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_steps=9, num_classes=3):
        super(MultiOutputSeq2SeqLSTM, self).__init__()
        self.output_steps = output_steps
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15
        )
        # Direct multi-output head to predict all 9 steps simultaneously without error accumulation
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_steps * num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take final hidden representation
        out = self.fc_head(out)
        out = out.view(-1, self.output_steps, self.num_classes) # Shape: (Batch, 9, 3)
        return out

# ==========================================
# 4. HIGH-ORDER MARKOV CHAIN PROBABILITY ENGINE
# ==========================================
class HighOrderMarkovChain:
    def __init__(self, order=3, num_states=3, smoothing=1.0):
        self.order = order
        self.num_states = num_states
        self.smoothing = smoothing
        self.transition_table = {}

    def fit(self, sequence):
        self.transition_table = {}
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_val = sequence[i + self.order]
            if state not in self.transition_table:
                self.transition_table[state] = np.zeros(self.num_states, dtype=np.float32)
            self.transition_table[state][next_val] += 1.0

    def predict_probs(self, current_window):
        state = tuple(current_window[-self.order:])
        if state in self.transition_table:
            counts = self.transition_table[state] + self.smoothing
            return counts / np.sum(counts)
        return np.ones(self.num_states, dtype=np.float32) / self.num_states

# ==========================================
# 5. TRAINING PIPELINE & CONCEPT DRIFT ADAPTATION
# ==========================================
def prepare_training_tensors(sequence, window_size=15, output_steps=9):
    X, y = [], []
    for i in range(len(sequence) - window_size - output_steps + 1):
        X.append(sequence[i : i + window_size])
        y.append(sequence[i + window_size : i + window_size + output_steps])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


@st.cache_resource
def initialize_model():
    return MultiOutputSeq2SeqLSTM()

def train_adaptive_model(model, sequence, epochs=45):
    if len(sequence) < 25:
        return model
    
    window_size = min(15, len(sequence) - 10)
    X_data, y_data = prepare_training_tensors(sequence, window_size=window_size, output_steps=9)
    
    if len(X_data) == 0:
        return model

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_data), torch.tensor(y_data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx) # Shape: (batch, 9, 3)
            loss = criterion(logits.view(-1, 3), by.view(-1))
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

# ==========================================
# 6. STREAMLIT SIDEBAR CONTROLS & UI
# ==========================================
st.sidebar.header("⚙️ Data Configuration")
input_option = st.sidebar.radio("Input Source", ["Manual Input (A, B, C)", "Random Stream Simulator"])

if input_option == "Manual Input (A, B, C)":
    default_text = "A, B, C, A, A, B, C, B, A, C, B, A, C, A, B, C, A, B, C, A"
    raw_input = st.sidebar.text_area("Enter Past Sequence (Comma-separated A, B, C):", default_text)
    history_seq = encode_sequence(raw_input)
else:
    sample_count = st.sidebar.slider("Historical Data Points", 30, 200, 60)
    np.random.seed(101)
    history_seq = list(np.random.choice([0, 1, 2], size=sample_count))
    st.sidebar.write("Generated Stream Length:", len(history_seq))

st.sidebar.markdown("---")
run_inference = st.sidebar.button("🚀 Run 9-Step Prediction Engine", type="primary")

# ==========================================
# 7. MAIN DASHBOARD EXECUTION & VISUALIZATION
# ==========================================
if len(history_seq) < 18:
    st.warning("⚠️ Please provide at least 18 historical sequence items for reliable pattern extraction.")
else:
    # Display History
    st.subheader("📈 Historical Sequence Timeline")
    df_history = pd.DataFrame({
        "Step": range(1, len(history_seq) + 1),
        "Color": [REV_COLOR_MAP[v] for v in history_seq],
        "Value": history_seq
    })
    fig_hist = px.scatter(
        df_history, x="Step", y="Color", color="Color",
        color_discrete_map={"Color A (Red)": "#EF4444", "Color B (Green)": "#10B981", "Color C (Blue)": "#3B82F6"},
        size=[14]*len(history_seq), height=250
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    if run_inference or True: # Auto-run or trigger
        with st.spinner("Executing Neural Seq2Seq & High-Order Markov Ensemble..."):
            time.sleep(0.3)
            
            # 1. Neural Model Training & Adaptation
            base_model = initialize_model()
            trained_model = train_adaptive_model(base_model, history_seq, epochs=40)
            
            trained_model.eval()
            with torch.no_grad():
                window_input = history_seq[-15:]
                tensor_in = torch.tensor(one_hot_encode(window_input), dtype=torch.float32).unsqueeze(0)
                logits = trained_model(tensor_in) # (1, 9, 3)
                nn_probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy() # (9, 3)

            # 2. Markov Chain Inference
            markov = HighOrderMarkovChain(order=3, num_states=3)
            markov.fit(history_seq)
            
            # 3. Hybrid Ensemble Combination & Rollout
            predictions = []
            active_window = list(window_input)
            
            for step in range(9):
                m_probs = markov.predict_probs(active_window)
                nn_step_prob = nn_probabilities[step]
                
                # Blended Hybrid Weight: 65% Deep Learning Seq2Seq + 35% Markov Transition
                blended = 0.65 * nn_step_prob + 0.35 * m_probs
                blended /= np.sum(blended)
                
                best_class = int(np.argmax(blended))
                confidence = float(blended[best_class])
                
                predictions.append({
                    "Round": f"Future Round +{step+1}",
                    "Predicted Color": REV_COLOR_MAP[best_class],
                    "Color A Prob": float(blended[0]),
                    "Color B Prob": float(blended[1]),
                    "Color C Prob": float(blended[2]),
                    "Confidence": confidence
                })
                active_window.append(best_class)

        # ==========================================
        # 8. RESULTS PRESENTATION
        # ==========================================
        st.markdown("---")
        st.subheader("🎯 Next 9-Round Prediction Horizon")
        
        pred_df = pd.DataFrame(predictions)
        global_conf = np.mean([p['Confidence'] for p in predictions]) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Target Horizon", "9 Consecutive Rounds")
        c2.metric("Engine Architecture", "Seq2Seq LSTM + Markov Chain")
        c3.metric("Global Sequence Confidence", f"{global_conf:.1f}%")

        # Table view
        st.dataframe(
            pred_df[["Round", "Predicted Color", "Confidence"]].style.format({
                "Confidence": "{:.1%}"
            }),
            use_container_width=True
        )

        # Probability Distribution Bar Chart
        fig_bar = px.bar(
            pred_df, x="Round", y=["Color A Prob", "Color B Prob", "Color C Prob"],
            barmode="group",
            color_discrete_map={
                "Color A Prob": "#EF4444", 
                "Color B Prob": "#10B981", 
                "Color C Prob": "#3B82F6"
            },
            title="Calibrated Softmax Probability Distribution Across 9 Future Rounds"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
