# -*- coding: utf-8 -*-
"""
DDoS Shield AI — Application Streamlit de détection d'attaques DDoS
===================================================================
Logique conservée du TP « Détection d'attaques DDoS avec Machine Learning » :
doublons -> NaN (fillna(0)) -> X / y -> split 70/15/15 -> standardisation
-> 5 modèles -> comparaison sur validation -> évaluation finale sur test.

NB : le jeu de données est entièrement SYNTHÉTIQUE (usage pédagogique).
Les scores obtenus ne démontrent pas une efficacité en production.
"""

import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
APP_NAME = "DDoS Shield AI"
DATA_PATH = Path(__file__).parent / "dataset.csv"

USERNAME = "maryem"
PASSWORD = "ddos2005"

NUM_COLS = [
    "duration_ms", "packet_count", "byte_count", "packets_per_second",
    "bytes_per_second", "avg_packet_size_bytes", "syn_count", "ack_count",
    "connections_10s", "unique_sources_10s",
]
CAT_COLS = ["protocol", "dst_port"]
DROP_COLS = ["label", "timestamp", "src_ip", "dst_ip", "src_port"]
TREE_MODELS = ("Random Forest", "XGBoost")

LABELS = {
    "duration_ms": "Durée (ms)",
    "packet_count": "Nombre de paquets",
    "byte_count": "Nombre d'octets",
    "packets_per_second": "Paquets / seconde",
    "bytes_per_second": "Octets / seconde",
    "avg_packet_size_bytes": "Taille moyenne paquet (octets)",
    "syn_count": "Paquets SYN",
    "ack_count": "Paquets ACK",
    "connections_10s": "Connexions (10 s)",
    "unique_sources_10s": "Sources uniques (10 s)",
}

C_ACCENT, C_OK, C_DANGER, C_MUTED = "#22d3ee", "#10b981", "#ef4444", "#94a3b8"

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")


# ----------------------------------------------------------------------------
# Logo + style
# ----------------------------------------------------------------------------
def logo_svg(size: int = 56) -> str:
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#6366f1"/>
    </linearGradient>
  </defs>
  <path d="M32 3 L57 12 V32 C57 47 46 57 32 61 C18 57 7 47 7 32 V12 Z"
        fill="#0f172a" stroke="url(#g)" stroke-width="3"/>
  <g stroke="url(#g)" stroke-width="2" fill="none" stroke-linecap="round">
    <line x1="32" y1="30" x2="20" y2="20"/><line x1="32" y1="30" x2="44" y2="20"/>
    <line x1="32" y1="30" x2="32" y2="47"/><line x1="32" y1="30" x2="19" y2="40"/>
    <line x1="32" y1="30" x2="45" y2="40"/>
  </g>
  <g fill="url(#g)">
    <circle cx="32" cy="30" r="5"/><circle cx="20" cy="20" r="3"/>
    <circle cx="44" cy="20" r="3"/><circle cx="32" cy="47" r="3"/>
    <circle cx="19" cy="40" r="3"/><circle cx="45" cy="40" r="3"/>
  </g>
</svg>"""


CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, #0b1220 55%, #070b14 100%);}
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
section[data-testid="stSidebar"] {background: #0a101d; border-right: 1px solid #1b2740;}
.brand {display:flex; align-items:center; gap:14px; margin-bottom: 6px;}
.brand .t {font-size: 1.55rem; font-weight: 800; line-height: 1.1;
  background: linear-gradient(90deg,#22d3ee,#818cf8); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent;}
.brand .s {color:#94a3b8; font-size:.85rem;}
.card {background: linear-gradient(180deg,#111a2e,#0e1627); border:1px solid #1f2a44;
  border-radius:16px; padding:18px 20px; box-shadow:0 8px 24px rgba(0,0,0,.25);}
.kpi .v {font-size:1.9rem; font-weight:800; color:#e2e8f0; line-height:1.1;}
.kpi .l {font-size:.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em;}
.kpi .d {font-size:.78rem; color:#22d3ee; margin-top:4px;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem;
  background:#12304a; color:#67e8f9; border:1px solid #1e4a6b;}
.banner {border-radius:18px; padding:26px 28px; display:flex; align-items:center; gap:18px;}
.banner.ok {background:linear-gradient(135deg,#052e22,#0b3d2e); border:1px solid #10b981;}
.banner.bad {background:linear-gradient(135deg,#3b0d12,#4c1117); border:1px solid #ef4444;}
.banner .big {font-size:2rem; font-weight:800;}
.banner .sm {color:#cbd5e1;}
.login-wrap {text-align:center; margin-top: 4vh;}
.login-wrap .brand {justify-content:center; flex-direction:column; gap:8px;}
.note {border-left:3px solid #f59e0b; background:#1c1710; padding:10px 14px;
  border-radius:8px; color:#fcd9a0; font-size:.85rem;}
div[data-testid="stForm"] {background:#0f172a; border:1px solid #1f2a44; border-radius:16px;}
.stButton>button, .stFormSubmitButton>button {border-radius:10px; font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def brand(size=52, subtitle="Détection d'attaques DDoS par Machine Learning"):
    st.markdown(
        f'<div class="brand">{logo_svg(size)}<div><div class="t">{APP_NAME}</div>'
        f'<div class="s">{subtitle}</div></div></div>', unsafe_allow_html=True)


def kpi(label, value, delta=""):
    st.markdown(f'<div class="card kpi"><div class="l">{label}</div>'
                f'<div class="v">{value}</div><div class="d">{delta}</div></div>',
                unsafe_allow_html=True)


def style_fig(fig, height=340):
    fig.update_layout(template="plotly_dark", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color="#cbd5e1"))
    return fig


# ----------------------------------------------------------------------------
# Authentification
# ----------------------------------------------------------------------------
def check_credentials(user: str, pwd: str) -> bool:
    ok_user = hmac.compare_digest(user.strip().lower().encode(), USERNAME.encode())
    ok_pwd = hmac.compare_digest(pwd.encode(), PASSWORD.encode())
    return ok_user and ok_pwd


def login_page():
    st.markdown("<style>section[data-testid='stSidebar'],"
                "div[data-testid='stSidebarCollapsedControl']{display:none;}</style>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown(
            f'<div class="login-wrap"><div class="brand">{logo_svg(84)}'
            f'<div class="t" style="font-size:2rem">{APP_NAME}</div>'
            f'<div class="s">Plateforme de détection d\'attaques DDoS</div></div></div>',
            unsafe_allow_html=True)
        st.write("")
        with st.form("login"):
            st.subheader("🔐 Connexion")
            user = st.text_input("Nom d'utilisateur", placeholder="Nom d'utilisateur")
            pwd = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            go_ = st.form_submit_button("Se connecter", type="primary", width="stretch")
        if go_:
            if check_credentials(user, pwd):
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")
        st.caption("ENSA Safi · Génie des Réseaux et Télécommunications")


# ----------------------------------------------------------------------------
# Données + entraînement (logique du TP conservée)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare(df_raw: pd.DataFrame, columns) -> pd.DataFrame:
    """Prétraitement identique à l'entraînement (pour la prédiction)."""
    d = df_raw[NUM_COLS + CAT_COLS].copy().fillna(0)
    d = pd.get_dummies(d, columns=CAT_COLS, dtype=int)
    return d.reindex(columns=columns, fill_value=0)


@st.cache_resource(show_spinner="Entraînement des 5 modèles en cours…")
def train_all():
    raw = load_raw()

    # 05 · doublons
    n_dup = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    # 06/07 · NaN -> fillna(0)
    n_nan = int(df.isna().sum().sum())
    df = df.fillna(0)

    # 08 · X / y
    y = df["label"]
    X = df.drop(columns=DROP_COLS)
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)

    # 09 · 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 10 · standardisation (fit sur train uniquement)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_val)
    Xte_s = scaler.transform(X_test)

    # 11-15 · modèles
    models, preds = {}, {}

    lin = LinearRegression().fit(Xtr_s, y_train)
    preds["Régression linéaire"] = (lin.predict(Xva_s) >= 0.5).astype(int)
    models["Régression linéaire"] = lin

    log = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, y_train)
    preds["Régression logistique"] = log.predict(Xva_s)
    models["Régression logistique"] = log

    svm = SVC(kernel="rbf", C=1.0).fit(Xtr_s, y_train)
    preds["SVM"] = svm.predict(Xva_s)
    models["SVM"] = svm

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
    rf.fit(X_train, y_train)
    preds["Random Forest"] = rf.predict(X_val)
    models["Random Forest"] = rf

    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                        eval_metric="logloss", random_state=42, n_jobs=2)
    xgb.fit(X_train, y_train)
    preds["XGBoost"] = xgb.predict(X_val)
    models["XGBoost"] = xgb

    # 16 · comparaison sur validation
    rows = []
    for nom, yp in preds.items():
        rows.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, yp),
            "Précision": precision_score(y_val, yp, zero_division=0),
            "Recall": recall_score(y_val, yp, zero_division=0),
            "F1-score": f1_score(y_val, yp, zero_division=0),
        })
    table = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)

    # 18 · test final (meilleur modèle uniquement)
    best = table.iloc[0]["Modèle"]
    Xte = X_test if best in TREE_MODELS else Xte_s
    pred_test = models[best].predict(Xte)
    if best == "Régression linéaire":
        pred_test = (pred_test >= 0.5).astype(int)

    test = {
        "Accuracy": accuracy_score(y_test, pred_test),
        "Précision": precision_score(y_test, pred_test, zero_division=0),
        "Recall": recall_score(y_test, pred_test, zero_division=0),
        "F1-score": f1_score(y_test, pred_test, zero_division=0),
        "cm": confusion_matrix(y_test, pred_test),
        "report": classification_report(y_test, pred_test,
                                        target_names=["Normal", "DDoS"], output_dict=True),
    }

    # Valeurs par défaut pour les exemples du formulaire (médianes par classe)
    presets = {}
    for lab, name in [(0, "normal"), (1, "ddos")]:
        sub = df[df["label"] == lab]
        presets[name] = {
            **{c: float(sub[c].median()) for c in NUM_COLS},
            "protocol": sub["protocol"].mode()[0],
            "dst_port": int(sub["dst_port"].mode()[0]),
        }

    return {
        "models": models, "scaler": scaler, "columns": list(X.columns),
        "table": table, "best": best, "test": test, "presets": presets,
        "sizes": (len(X_train), len(X_val), len(X_test)),
        "n_raw": len(raw), "n_dup": n_dup, "n_nan": n_nan, "n_clean": len(df),
        "importances": pd.Series(rf.feature_importances_, index=X.columns)
                         .sort_values(ascending=False),
    }


def predict_rows(art, model_name: str, X: pd.DataFrame):
    """Retourne (étiquettes 0/1, score DDoS ou None)."""
    m = art["models"][model_name]
    Xin = X if model_name in TREE_MODELS else art["scaler"].transform(X)
    if model_name == "Régression linéaire":
        s = np.clip(m.predict(Xin), 0, 1)
        return (s >= 0.5).astype(int), s
    labels = m.predict(Xin)
    score = m.predict_proba(Xin)[:, 1] if hasattr(m, "predict_proba") else None
    return labels, score


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard(art):
    raw = load_raw()
    st.subheader("📊 Tableau de bord — Statistiques du dataset")
    st.markdown('<div class="note">Jeu de données <b>entièrement synthétique</b> (logs agrégés '
                'de flux réseau, usage pédagogique) : les scores ne prouvent pas une efficacité '
                'en production.</div>', unsafe_allow_html=True)
    st.write("")

    pct = raw["label"].mean() * 100
    c = st.columns(5)
    with c[0]: kpi("Lignes", f"{art['n_raw']:,}".replace(",", " "), f"{art['n_clean']:,} après nettoyage".replace(",", " "))
    with c[1]: kpi("Colonnes", raw.shape[1], "dont 1 cible (label)")
    with c[2]: kpi("Doublons", art["n_dup"], "supprimés")
    with c[3]: kpi("Valeurs manquantes", art["n_nan"], "remplacées par 0")
    with c[4]: kpi("Part DDoS", f"{pct:.1f} %", f"Normal : {100 - pct:.1f} %")
    st.write("")

    a, b = st.columns(2)
    with a:
        counts = raw["label"].map({0: "Normal", 1: "DDoS"}).value_counts().reset_index()
        counts.columns = ["Classe", "n"]
        fig = px.pie(counts, names="Classe", values="n", hole=0.62, color="Classe",
                     color_discrete_map={"Normal": C_OK, "DDoS": C_DANGER},
                     title="Répartition des classes")
        st.plotly_chart(style_fig(fig), width="stretch")
    with b:
        proto = (raw.assign(Classe=raw["label"].map({0: "Normal", 1: "DDoS"}))
                 .groupby(["protocol", "Classe"]).size().reset_index(name="n"))
        fig = px.bar(proto, x="protocol", y="n", color="Classe", barmode="group",
                     color_discrete_map={"Normal": C_OK, "DDoS": C_DANGER},
                     title="Protocoles par classe", labels={"protocol": "Protocole", "n": "Flux"})
        st.plotly_chart(style_fig(fig), width="stretch")

    a, b = st.columns(2)
    with a:
        feat = st.selectbox("Distribution d'une variable (échelle log)",
                            NUM_COLS, index=3, format_func=LABELS.get)
        d = raw[[feat, "label"]].dropna()
        d = d[d[feat] > 0].assign(Classe=lambda x: x["label"].map({0: "Normal", 1: "DDoS"}))
        fig = px.histogram(d, x=feat, color="Classe", nbins=60, log_x=True, barmode="overlay",
                           opacity=0.65, color_discrete_map={"Normal": C_OK, "DDoS": C_DANGER})
        st.plotly_chart(style_fig(fig, 320), width="stretch")
    with b:
        t = raw.assign(jour=pd.to_datetime(raw["timestamp"]).dt.date,
                       Classe=raw["label"].map({0: "Normal", 1: "DDoS"}))
        t = t.groupby(["jour", "Classe"]).size().reset_index(name="n")
        fig = px.line(t, x="jour", y="n", color="Classe", markers=True,
                      title="Flux par jour", color_discrete_map={"Normal": C_OK, "DDoS": C_DANGER},
                      labels={"jour": "Jour", "n": "Flux"})
        st.plotly_chart(style_fig(fig, 320), width="stretch")

    with st.expander("🔎 Aperçu des données (5 premières lignes) et statistiques descriptives"):
        st.dataframe(raw.head(), width="stretch")
        st.dataframe(raw.describe().T.round(2), width="stretch")
        st.caption(f"Colonnes : {', '.join(raw.columns)}")


def page_models(art):
    st.subheader("🧠 Modèles & métriques")
    tr, va, te = art["sizes"]
    st.caption(f"Découpage stratifié 70 / 15 / 15 → Train : {tr} · Validation : {va} · Test : {te}")

    t = art["table"]
    st.markdown("#### Comparaison sur l'ensemble de validation")
    st.dataframe(
        t.style.format({k: "{:.3f}" for k in t.columns[1:]})
         .highlight_max(subset=t.columns[1:], color="#0e4f5c")
         .set_properties(**{"font-weight": "600"}, subset=["Modèle"]),
        width="stretch", hide_index=True)

    melt = t.melt(id_vars="Modèle", var_name="Métrique", value_name="Score")
    fig = px.bar(melt, x="Modèle", y="Score", color="Métrique", barmode="group",
                 color_discrete_sequence=["#22d3ee", "#818cf8", "#f472b6", "#fbbf24"])
    fig.update_yaxes(range=[0, 1.02])
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    best = art["best"]
    st.markdown(f"#### 🏆 Meilleur modèle (F1 validation) : <span class='pill'>{best}</span>",
                unsafe_allow_html=True)
    st.caption("Évaluation finale sur l'ensemble de test, jamais utilisé pour choisir le modèle.")
    tst = art["test"]
    c = st.columns(4)
    for col, k in zip(c, ["Accuracy", "Précision", "Recall", "F1-score"]):
        with col: kpi(k, f"{tst[k]:.3f}", "jeu de test")
    st.write("")

    a, b = st.columns(2)
    with a:
        cm = tst["cm"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Normal", "DDoS"], y=["Normal", "DDoS"], colorscale="Blues",
            text=cm, texttemplate="%{text}", textfont=dict(size=22), showscale=False))
        fig.update_layout(title="Matrice de confusion — test final",
                          xaxis_title="Prédit", yaxis_title="Réel", yaxis_autorange="reversed")
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        fn, fp = int(cm[1, 0]), int(cm[0, 1])
        st.caption(f"Faux négatifs (attaques manquées) : **{fn}** · "
                   f"Faux positifs (fausses alertes) : **{fp}**")
    with b:
        imp = art["importances"].head(12)[::-1]
        fig = px.bar(imp, x=imp.values, y=imp.index, orientation="h",
                     title="Variables les plus importantes (Random Forest)",
                     labels={"x": "Importance", "y": ""})
        fig.update_traces(marker_color=C_ACCENT)
        st.plotly_chart(style_fig(fig, 360), width="stretch")

    with st.expander("Rapport de classification détaillé"):
        st.dataframe(pd.DataFrame(tst["report"]).T.round(3), width="stretch")
    st.info("La régression linéaire est une baseline pédagogique : ce n'est pas un vrai "
            "classifieur probabiliste (la régression logistique est plus adaptée).")


def apply_preset(art, which: str):
    p = art["presets"][which]
    for c in NUM_COLS:
        st.session_state[f"f_{c}"] = round(p[c], 2)
    st.session_state["f_protocol"] = p["protocol"]
    st.session_state["f_dst_port"] = p["dst_port"]


def result_banner(label: int, score):
    if label == 1:
        cls, icon, title, sub = "bad", "🚨", "ATTAQUE DDoS DÉTECTÉE", "Ce flux présente un comportement malveillant."
    else:
        cls, icon, title, sub = "ok", "✅", "TRAFIC NORMAL", "Aucun comportement d'attaque détecté."
    extra = f" · Score DDoS : <b>{score * 100:.1f} %</b>" if score is not None else ""
    st.markdown(f'<div class="banner {cls}"><div style="font-size:3rem">{icon}</div>'
                f'<div><div class="big">{title}</div><div class="sm">{sub}{extra}</div></div></div>',
                unsafe_allow_html=True)


def page_predict(art):
    st.subheader("🎯 Prédiction : Normal ou DDoS")
    models = list(art["models"].keys())
    model_name = st.selectbox("Modèle utilisé", models, index=models.index(art["best"]),
                              help="Par défaut : le meilleur modèle selon le F1-score de validation.")
    tab1, tab2 = st.tabs(["✍️ Saisie manuelle", "📁 Fichier CSV (par lot)"])

    with tab1:
        if "f_protocol" not in st.session_state:
            apply_preset(art, "normal")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.button("Exemple trafic normal", on_click=apply_preset, args=(art, "normal"), width="stretch")
        b2.button("Exemple attaque DDoS", on_click=apply_preset, args=(art, "ddos"), width="stretch")

        with st.form("manual"):
            c1, c2 = st.columns(2)
            c1.selectbox("Protocole", ["TCP", "UDP", "ICMP"], key="f_protocol")
            c2.selectbox("Port de destination", [0, 22, 53, 80, 123, 443, 8080], key="f_dst_port")
            cols = st.columns(3)
            for i, c in enumerate(NUM_COLS):
                cols[i % 3].number_input(LABELS[c], min_value=0.0, step=1.0, key=f"f_{c}")
            run = st.form_submit_button("🔍 Lancer la prédiction", type="primary", width="stretch")

        if run:
            row = pd.DataFrame([{**{c: st.session_state[f"f_{c}"] for c in NUM_COLS},
                                 "protocol": st.session_state["f_protocol"],
                                 "dst_port": st.session_state["f_dst_port"]}])
            X = prepare(row, art["columns"])
            lab, score = predict_rows(art, model_name, X)
            result_banner(int(lab[0]), None if score is None else float(score[0]))
            if score is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(score[0]) * 100,
                    number={"suffix": " %"}, title={"text": "Probabilité d'attaque DDoS"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": C_DANGER if lab[0] else C_OK},
                           "steps": [{"range": [0, 50], "color": "#0f2a22"},
                                     {"range": [50, 100], "color": "#2d1015"}]}))
                st.plotly_chart(style_fig(fig, 260), width="stretch")
            else:
                st.caption("Ce modèle (SVM) ne fournit pas de probabilité : seul le verdict est affiché.")

    with tab2:
        st.caption("Chargez un CSV contenant au minimum les colonnes : "
                   + ", ".join(NUM_COLS + CAT_COLS))
        up = st.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            try:
                d = pd.read_csv(up)
                missing = [c for c in NUM_COLS + CAT_COLS if c not in d.columns]
                if missing:
                    st.error(f"Colonnes manquantes : {', '.join(missing)}")
                else:
                    lab, score = predict_rows(art, model_name, prepare(d, art["columns"]))
                    out = d.copy()
                    out["prediction"] = np.where(lab == 1, "DDoS", "Normal")
                    if score is not None:
                        out["score_ddos"] = score.round(4)
                    n1 = int((lab == 1).sum())
                    c = st.columns(3)
                    with c[0]: kpi("Flux analysés", len(out))
                    with c[1]: kpi("Détectés DDoS", n1, f"{n1 / len(out) * 100:.1f} %")
                    with c[2]: kpi("Normaux", len(out) - n1)
                    st.write("")
                    st.dataframe(out.head(200), width="stretch")
                    st.download_button("⬇️ Télécharger les résultats (CSV)",
                                       out.to_csv(index=False).encode("utf-8"),
                                       "predictions_ddos.csv", "text/csv")
            except Exception as e:  # fichier illisible
                st.error(f"Impossible de lire le fichier : {e}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not st.session_state.get("auth"):
        login_page()
        return

    art = train_all()

    with st.sidebar:
        st.markdown(f'<div class="brand">{logo_svg(44)}<div><div class="t" style="font-size:1.2rem">'
                    f'{APP_NAME}</div><div class="s">v1.0</div></div></div>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigation", ["📊 Tableau de bord", "🧠 Modèles & métriques",
                                       "🎯 Prédiction"], label_visibility="collapsed")
        st.divider()
        st.caption(f"👤 Connecté : **{USERNAME.capitalize()}**")
        if st.button("Se déconnecter", width="stretch"):
            st.session_state.clear()
            st.rerun()

    brand()
    st.write("")
    {"📊 Tableau de bord": page_dashboard,
     "🧠 Modèles & métriques": page_models,
     "🎯 Prédiction": page_predict}[page](art)


main()
