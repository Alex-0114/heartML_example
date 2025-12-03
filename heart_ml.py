# heart_ml_app.py
"""
Heart Disease – ML Demo

Piccola applicazione Streamlit che:
- carica un dataset pubblico su malattia cardiaca
- allena un modello binario (malattia sì/no)
- mostra alcune metriche di performance
- permette di fare una previsione per un singolo paziente
"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# *PAGE CONFIG 
st.set_page_config(page_title="Heart Disease – ML Demo", layout="wide", page_icon="❤️")

# *VARS  -----------------------------------------------------------
# Data path
DATA_PATH = Path("data/heart.csv")

# Cols selezionate come feature
FEATURE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Etichette leggibili in italiano per l'UI
FEATURE_LABELS = {
    "age": "Età (anni)",
    "trestbps": "Pressione a riposo (mm Hg)",
    "chol": "Colesterolo (mg/dl)",
    "thalch": "Freq. cardiaca max (bpm)",
    "oldpeak": "Depressione ST (oldpeak)",
}
TARGET_LABEL = "Presenza di malattia (target)"

# *UTILS -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Legge il csv e crea la colonna target binaria."""
    df = pd.read_csv(DATA_PATH)

    # num: 0 = sano, 1–4 = malattia
    df["target"] = (df["num"] > 0).astype(int)

    cols = FEATURE_COLS + ["target"]
    return df[cols]


@st.cache_resource
def train_model(df: pd.DataFrame, depth: int, samples_leaf: int):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth = depth,
        min_samples_leaf = samples_leaf,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # baseline: predire sempre la classe più frequente
    majority_class = int(y_test.value_counts().idxmax())
    baseline_pred = [majority_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred) # calcola acc se y_pred fosse sempre 1 = malato contro y_true

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # importanza delle feature
    fi = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    ).sort_values(ascending=True)

    return model, metrics, fi


######################################################################
# ------------------------------ ST APP ---------------------------- #
######################################################################

df = load_data()

st.title("❤️ Heart Disease – ML Demo")
st.caption(
    "Esempio didattico: modello binario (malattia sì/no) su 5 feature "
    "numeriche. Non è uno strumento medico reale."
)

# *PANORAMICA --------------------------------------------------------

st.subheader("🔍 Panoramica del dataset")

col_a, col_b, col_c = st.columns(3)

n_patients = len(df)
positive_rate = df["target"].mean()

col_a.metric("Numero pazienti", n_patients)
col_b.metric("Con malattia (%)", f"{positive_rate:.1%}")
col_c.metric(
    "Sani vs malati",
    f"{(1 - positive_rate):.1%} sani / {positive_rate:.1%} malati",
)

with st.expander("Esplora la distribuzione delle variabili"):
    variabileUtente = st.selectbox("Seleziona variabile da visualizzare", FEATURE_LABELS.values())
    col_name = [chiave for chiave, valore in FEATURE_LABELS.items() if valore == variabileUtente]
    
    left, right = st.columns(2)
    with left:
        fig_hist, ax_hist = plt.subplots()
        
        sns.histplot(data=df, x=col_name[0], kde=True, ax=ax_hist)
        ax_hist.set_title(f"Distribuzione di {variabileUtente}")
        ax_hist.set_xlabel(variabileUtente)
        ax_hist.set_ylabel("Frequenza")
        st.pyplot(fig_hist)

    with right:
        fig_violin, ax_violin = plt.subplots()
        sns.violinplot(data=df, x="target", y=col_name[0], ax=ax_violin, palette="pastel")
        ax_violin.set_title(f"{variabileUtente} vs Presenza di malattia")
        ax_violin.set_xticklabels(["Sano (0)", "Malato (1)"])
        ax_violin.set_ylabel(variabileUtente)
        st.pyplot(fig_violin)


with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())


#SIDEBAR
st.sidebar.header("Parametri modello")
st.sidebar.markdown("---")

model_depth = st.sidebar.slider(
    "Profondità massima degli alberi",
    min_value=1,
    max_value=20,
    value=4,
)

min_sample = st.sidebar.slider(
    "Minimo campioni per foglia",
    min_value=1,
    max_value=20,
    value=4,
)

# * RF PERFORMANCE ---------------------------------------------------

model, metrics, feature_importances = train_model(df, model_depth, min_sample)

st.subheader("📏 Performance del modello")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")
col3.metric(
    "Baseline (classe più frequente)",
    f"{metrics['baseline_acc']:.2%}",
)

st.caption(
    "Se l'accuracy su test è simile a quella su train e migliore della "
    "baseline, il modello sta generalizzando in modo ragionevole."
)

# *CORR & FEATURE IMPORTANCE -----------------------------------------

st.subheader("📈 Correlazioni e importanza delle variabili")

col_corr, col_imp = st.columns(2)

with col_corr:
    st.markdown("**Correlazione tra variabili e target**")

    # Rename cols for corr matrix plot
    df_corr = df.copy()
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    rename_map["target"] = TARGET_LABEL
    df_corr = df_corr.rename(columns=rename_map)

    # compute corr matrix
    corr = df_corr.corr()

    # corr matrix heatmap with sns
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax_corr,
    )
    ax_corr.set_title("Matrice di correlazione")
    st.pyplot(fig_corr)

    # Scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
    target_corr = df_corr.corr()[TARGET_LABEL].drop(TARGET_LABEL)
    ax_scatter.scatter(target_corr, feature_importances, s=70, alpha=0.6, color='red')
    for i, label in enumerate(feature_importances.index):
        ax_scatter.annotate(label, (target_corr.iloc[i], feature_importances.iloc[i]), fontsize=9)
    ax_scatter.set_xlabel("Correlazione con target")
    ax_scatter.set_ylabel("Importanza")
    ax_scatter.set_title("Feature: Correlazione e Importanza")
    plt.tight_layout()
    st.pyplot(fig_scatter)

with col_imp:
    st.markdown("**Importanza delle variabili (RandomForest)**")

    # Plot feature importances barchart with matplotlib
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    feature_importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_xlabel("Importanza (Gini)")
    ax_imp.set_ylabel("Variabile")
    ax_imp.set_title("Importanza delle variabili")
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.caption(f"La variabile più importante è **{feature_importances.idxmax()}**.")


# *FORM PAZIENTE ----------------------------------------------------

st.subheader("🧪 Inserisci i dati del paziente")

cols = st.columns(3)
user_input: dict[str, float] = {}

for i, col_name in enumerate(FEATURE_COLS):
    serie = df[col_name]
    min_val = float(serie.min())
    max_val = float(serie.max())
    default = float(serie.median()) # settiamo il valore di deafault sulla mediana

    label = FEATURE_LABELS[col_name]

    with cols[i % 3]: # <---- watch out
        # prenndiamo lo user input con number input di streamlit
        user_input[col_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )

if st.button("Predici rischio"):
    input_df = pd.DataFrame([user_input])
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] > 0.5)

    col_res1, col_res2 = st.columns(2)
    label_risk = "ALTO" if pred == 1 else "BASSO"
    col_res1.metric("Rischio stimato", label_risk)
    col_res2.metric("Probabilità di malattia", f"{proba[1]:.1%}")

    st.write("Valori inseriti:")
    pretty_input = {FEATURE_LABELS[k]: v for k, v in user_input.items()}
    st.json(pretty_input)

    st.info(
        "⚠️ Esempio didattico su un dataset pubblico. "
        "Non è uno strumento clinico e non va usato per decisioni reali."
    )