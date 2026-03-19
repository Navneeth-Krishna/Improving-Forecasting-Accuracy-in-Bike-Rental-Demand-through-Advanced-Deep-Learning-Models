import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, time
import altair as alt
from datetime import timedelta


DATA_FILE = "SeoulBikeData.csv"

MODEL_PATHS = {
    "Temporal CNN": "saved_models_new/temporal_cnn",
    "ConvLSTM": "saved_models_new/convlstm",
    "LSTM + GRU + Attention": "saved_models_new/lstm_gru_attention",
    "Transformer": "saved_models_new/transformer",
    "LSTM + Random Forest": "saved_models_new/lstm_rf",
}

TARGET_COL = "Rented Bike Count"

INPUT_FEATURES = 15

# MODEL DEFINITIONS

class TemporalCNN(nn.Module):
    def __init__(self, input_size, channels=80, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.bn(x)
        x = torch.relu(x)
        avg_pool = x.mean(dim=2)
        max_pool, _ = x.max(dim=2)
        feats = torch.cat([avg_pool, max_pool], dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)


class ConvLSTM(nn.Module):
    def __init__(self, input_size, conv_channels=40, lstm_hidden=80, num_lstm_layers=2, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, 3, padding=1)
        self.conv_bn = nn.BatchNorm1d(conv_channels)
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv_bn(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        final_h = h_n[-1]
        final_h = self.dropout(final_h)
        return self.fc(final_h)


class DeepLSTMGRUAttention(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.attn_fc = nn.Linear(32, 32)
        self.attn_score = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.gru(x)
        attn_weights = torch.tanh(self.attn_fc(x))
        attn_weights = torch.softmax(self.attn_score(attn_weights), dim=1)
        x = (x * attn_weights).sum(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, dropout=0.25):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out_last = out[:, -1, :]
        out_last = self.dropout(out_last)
        return self.fc(out_last)

    def extract_features(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        feats = out[:, -1, :]
        return feats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        last_t = x[:, -1, :]
        return self.fc_out(last_t)


# DATA PREP

def load_data():
    df = pd.read_csv(DATA_FILE, encoding="latin1")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    return df


def add_engineered_features(df, max_lag=3):
    df = df.copy()
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["is_morning_peak"] = df["Hour"].isin([7, 8, 9, 10]).astype(int)
    df["is_evening_peak"] = df["Hour"].isin([17, 18, 19, 20]).astype(int)
    df["is_low_demand_hour"] = df["Hour"].between(0, 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["Holiday"] = df["Holiday"].map({"No Holiday": 0, "Holiday": 1})
    df["Functioning Day"] = df["Functioning Day"].map({"Yes": 1, "No": 0})

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    if "Seasons" in df.columns:
        df = pd.get_dummies(df, columns=["Seasons"], drop_first=True)

    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df):
    unique_dates = df["Date"].dt.date.unique()
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    train_df = df[df["Date"].dt.date.isin(train_dates)].reset_index(drop=True)
    test_df = df[df["Date"].dt.date.isin(test_dates)].reset_index(drop=True)
    return train_df, test_df


def build_timestamp(df):
    df = df.copy()
    df["timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    return df


# MODEL LOADING

def instantiate_model(model_name, config):
    if model_name == "Temporal CNN":
        return TemporalCNN(
            input_size=config["input_size"],
            channels=config.get("channels", 80),
            dropout=config.get("dropout", 0.2),
        )

    elif model_name == "ConvLSTM":
        return ConvLSTM(
            input_size=config["input_size"],
            conv_channels=config.get("conv_channels", 40),
            lstm_hidden=config.get("lstm_hidden", 80),
            num_lstm_layers=config.get("num_lstm_layers", 2),
            dropout=config.get("dropout", 0.2),
        )

    elif model_name == "LSTM + GRU + Attention":
        return DeepLSTMGRUAttention(
            input_size=config["input_size"],
            dropout=config.get("dropout", 0.2),
        )

    elif model_name == "Transformer":
        return TimeSeriesTransformer(
            input_size=config["input_size"],
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 3),
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=config.get("dropout", 0.2),
        )

    elif model_name == "LSTM + Random Forest":
        return LSTMFeatureExtractor(
            input_dim=config["input_size"],
            hidden_dim1=config.get("hidden_dim1", 128),
            hidden_dim2=config.get("hidden_dim2", 64),
            hidden_dim3=config.get("hidden_dim3", 32),
            dropout=config.get("dropout", 0.25),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


@st.cache_resource
def load_model_bundle(model_name, model_dir):
    if not os.path.exists(model_dir):
        return None

    required_files = [
        "model_config.pkl",
        "scaler_X.pkl",
        "scaler_y.pkl",
        "selected_features.pkl",
        "sequence_length.pkl",
    ]

    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            return None

    config = joblib.load(os.path.join(model_dir, "model_config.pkl"))
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    selected_features = joblib.load(os.path.join(model_dir, "selected_features.pkl"))
    seq_len = joblib.load(os.path.join(model_dir, "sequence_length.pkl"))
    config["input_size"] = INPUT_FEATURES

    model = instantiate_model(model_name, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weight_path = os.path.join(model_dir, "model_weights.pth")
    if not os.path.exists(weight_path):
        return None

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rf_model = None
    if model_name == "LSTM + Random Forest":
        rf_path = os.path.join(model_dir, "rf_model.pkl")
        if not os.path.exists(rf_path):
            return None
        rf_model = joblib.load(rf_path)

    return {
        "model": model,
        "rf_model": rf_model,
        "config": config,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "selected_features": selected_features,
        "seq_len": seq_len,
        "device": device,
    }


# PREDICTION

def predict_one(model_name, bundle, window_df):
    selected_features = bundle["selected_features"]
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    device = bundle["device"]

    X_window = window_df[selected_features].values.astype(np.float32)
    X_scaled = scaler_X.transform(X_window)
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(0).to(device)

    if model_name == "LSTM + Random Forest":
        with torch.no_grad():
            feats = bundle["model"].extract_features(X_tensor).cpu().numpy()
        pred = bundle["rf_model"].predict(feats)[0]
        return int(round(pred))

    else:
        with torch.no_grad():
            pred_scaled = bundle["model"](X_tensor).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        return int(round(pred))


# APP

st.set_page_config(page_title="Bike Demand Demo", layout="wide")

st.title("Bike-Sharing Demand Forecast Demo")
st.markdown("""
Predict next-hour bike rental demand using multiple deep learning models.
Compare model predictions and visualize demand trends.
""")
st.divider()
# Load and preprocess data
df_raw = load_data()
df_eng = add_engineered_features(df_raw)
_, test_df = chronological_split(df_eng)
test_df = build_timestamp(test_df)

days_to_use = 15
unique_test_dates = pd.Series(test_df["Date"].dt.date.unique())
last_dates = unique_test_dates.iloc[-days_to_use:]
demo_df = test_df[test_df["Date"].dt.date.isin(last_dates)].reset_index(drop=True)

# models
available_models = []
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        available_models.append(name)


st.subheader("Select Model(s) for Prediction")

selected_models = []


cols = st.columns(len(available_models))

for i, model in enumerate(available_models):
    if cols[i].checkbox(model, value=(model == available_models[0])):
        selected_models.append(model)
if not selected_models:
    st.warning("Please choose at least one model.")
    st.stop()

#  date selector
available_dates = sorted(demo_df["Date"].dt.date.unique())
selected_date = st.date_input(
    "Select prediction date",
    value=available_dates[0],
    min_value=available_dates[0],
    max_value=available_dates[-1]
)

# Filter hours for selected date
date_df = demo_df[demo_df["Date"].dt.date == selected_date].copy()
available_hours = sorted(date_df["Hour"].unique().tolist())

selected_hour = st.selectbox(
    "Select prediction hour",
    options=available_hours,
    index=0,
    format_func=lambda x: f"{x}:00",
    help="This is the target hour to be predicted using the previous 24 hours."
)

predict_button = st.button("Predict Next-Hour Demand")

if predict_button:

    target_rows = test_df[
        (test_df["Date"].dt.date == selected_date) &
        (test_df["Hour"] == selected_hour)
    ]

    if len(target_rows) == 0:
        st.error("Selected date/hour not found in test set.")
        st.stop()

    target_idx = target_rows.index[0]

    seq_len = 24
    if target_idx < seq_len:
        st.error("Not enough historical observations before this timestamp.")
        st.stop()

    window_df = test_df.iloc[target_idx - seq_len: target_idx].copy()

    st.subheader("Weather Conditions at Prediction Time")

    weather_cols = [
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Rainfall(mm)",
        "Snowfall (cm)"
    ]

    weather_data = test_df.loc[target_idx, weather_cols]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Temperature", f"{weather_data[0]} °C")
    col2.metric("Humidity", f"{weather_data[1]} %")
    col3.metric("Wind Speed", f"{weather_data[2]} m/s")
    col4.metric("Rainfall", f"{weather_data[3]} mm")
    col5.metric("Snowfall", f"{weather_data[4]} cm")
    actual_value = float(test_df.loc[target_idx, TARGET_COL])
    target_ts = test_df.loc[target_idx, "timestamp"]

    results = []

    for model_name in selected_models:
        bundle = load_model_bundle(model_name, MODEL_PATHS[model_name])

        if bundle is None:
            results.append({
                "Model": model_name,
                "Predicted Demand": "Bundle missing",
                "Actual Demand": int(actual_value),
                "Error": "N/A"
            })
            continue

        try:
            pred = predict_one(model_name, bundle, window_df)

            error = abs(pred - actual_value)

            results.append({
                "Model": model_name,
                "Predicted Demand": int(pred),
                "Actual Demand": int(actual_value),
                "Error": int(error)
            })

        except Exception as e:
            results.append({
                "Model": model_name,
                "Predicted Demand": f"Error: {str(e)}",
                "Actual Demand": int(actual_value),
                "Error": "N/A"
            })

    #  Results

    st.subheader(f"Prediction for {target_ts}")

    results_df = pd.DataFrame(results)

    # Find best model (lowest error)
    best_error = results_df["Error"].replace("N/A", np.nan).astype(float).min()

    def highlight_best(row):
        try:
            if float(row["Error"]) == best_error:
                return ["background-color: #173928; color: #38DA88; font-weight: bold"] * len(row)
        except:
            pass
        return [""] * len(row)

    styled_df = results_df.style.apply(highlight_best, axis=1)

    st.dataframe(styled_df, use_container_width=True, hide_index=True)


#   grpah

    st.subheader("Demand Forecast Visualization")

    history_plot = window_df.copy()
    history_plot["timestamp"] = history_plot["Date"] + pd.to_timedelta(history_plot["Hour"], unit="h")

    history_df = history_plot[["timestamp", TARGET_COL]].copy()
    history_df = history_df.rename(columns={TARGET_COL: "Demand"})

    pred_rows = []
    pred_rows.append({
        "timestamp": target_ts,
        "Demand": actual_value,
        "Model": "Actual"
    })

    for _, row in results_df.iterrows():
        if isinstance(row["Predicted Demand"], int):
            pred_rows.append({
                "timestamp": target_ts,
                "Demand": row["Predicted Demand"],
                "Model": row["Model"]
            })

    pred_df = pd.DataFrame(pred_rows)

    min_time = history_df["timestamp"].min()
    max_time = target_ts + timedelta(hours=3)

    x_axis = alt.X(
        "timestamp:T",
        scale=alt.Scale(domain=[min_time, max_time]),
        title="Time"
    )

    # History demand line
    line = alt.Chart(history_df).mark_line(
        point=True,
        color="#4C78A8"
    ).encode(
        x=x_axis,
        y=alt.Y("Demand:Q", title="Bike Demand"),
        tooltip=["timestamp:T", "Demand:Q"]
    )

    # Prediction points
    points = alt.Chart(pred_df).mark_circle(
        size=150
    ).encode(
        x=x_axis,
        y="Demand:Q",
        color=alt.Color("Model:N", title="Model"),
        tooltip=["Model:N", "Demand:Q"]
    )

    # Combine charts
    chart = (line + points).properties(
        width=800,
        height=400,
        title="Last 24 Hours Demand and Model Predictions"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


    st.info(
        "Prediction is generated using the previous 24 hours of observations from the test set. "
        "The target hour is chosen from the last selected days of the test period."
    )