import streamlit as st
import pandas as pd
import time
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import base64
import hashlib

# Page configuration
st.set_page_config(
    page_title="Troubleshooting Guide System",
    page_icon="🚄",
    layout="wide"
)

st.markdown("""
    <style>
        /* App background */
        .stApp {
            background-color: #01311F;
            color: #F0ECE3;
        }

        /* General text */
        html, body, .stApp {
            font-family: 'Segoe UI', sans-serif;
            color: #F0ECE3;
        }

        /* Title and large timer */
        .title, .big-timer {
            text-align: center;
            color: #F0ECE3;
            font-weight: bold;
        }

        .big-timer {
            font-size: 48px;
        }

        /* Buttons */
        .stButton>button {
            background-color: #F0ECE3;
            color: #01311F;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            margin: 8px 0;
            width: 100%;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #B99A49;
            color: #01311F;
            transform: scale(1.03);
        }

        /* Scenario buttons */
        .scenario-btn {
            background-color: #F0ECE3;
            color: #01311F;
            border: 2px solid #B99A49;
            border-radius: 10px;
            padding: 14px;
            font-weight: 500;
            margin: 10px 0;
        }

        .scenario-btn:hover {
            background-color: #B99A49;
            color: #F0ECE3;
        }

        /* Response / card sections */
        .response-section, .metric-card {
            background-color: #F0ECE3;
            color: #01311F;
            padding: 16px;
            border-radius: 8px;
        }

        /* Match highlights */
        .match-success {
            background-color: #B99A49;
            color: #01311F;
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }

        .match-fail {
            background-color: #FFE4E1;
            color: #01311F;
            text-align: center;
            padding: 10px;
            border-radius: 6px;
            font-weight: bold;
        }

        /* Status tags */
        .status-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
            min-width: 60px;
        }
        
        .status-major {
            background: #FF4D4D;
            color: #FFFFFF;
        }
        
        .status-minor {
            background: #FFA726;
            color: #000000;
        }
        
        /* Timer container */
        .timer-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
        }
        
        /* Center the stop timer button */
        .timer-container .stButton>button {
            max-width: 200px;
            margin-top: 20px;
        }
        
        /* Similarity gauges */
        .similarity-gauge {
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin: 10px 0;
            position: relative;
        }
        
        .gauge-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        .threshold-marker {
            position: absolute;
            top: -5px;
            width: 2px;
            height: 20px;
            background-color: #000;
        }
    </style>
""", unsafe_allow_html=True)

# --- Hybrid Model Setup (SBERT + TF-IDF + Cosine Similarity) ---
@st.cache_resource
def load_models():
    """Load both SBERT model and initialize TF-IDF vectorizer"""
    try:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        return sbert_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure required packages are installed: pip install sentence-transformers scikit-learn")
        return None, None

sbert_model, tfidf_vectorizer = load_models()
SBERT_THRESHOLD = 0.4
TFIDF_THRESHOLD = 0.15
HYBRID_WEIGHT_SBERT = 0.6
HYBRID_WEIGHT_TFIDF = 0.4

def preprocess_text_for_matching(text):
    """Enhanced text preprocessing for better matching"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to lowercase and remove extra whitespace
    text = str(text).lower().strip()
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\(\)\/]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_first_sentences(text, num_sentences=3):
    """Extract first N sentences for better matching on key information"""
    if not text or pd.isna(text):
        return ""
    
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', str(text))
    # Take first N sentences and clean them
    first_sentences = []
    for i, sentence in enumerate(sentences[:num_sentences]):
        sentence = sentence.strip()
        if sentence:  # Only add non-empty sentences
            first_sentences.append(sentence)
    
    return '. '.join(first_sentences) if first_sentences else str(text)

def hybrid_similarity_match(user_input, predefined_list):
    """
    Enhanced hybrid approach with better preprocessing and lower thresholds
    """
    if not sbert_model or not predefined_list or not user_input.strip():
        return None, None
    
    try:
        # Preprocess input
        processed_input = preprocess_text_for_matching(user_input)
        
        # Preprocess predefined list with first 3 sentences focus
        processed_list = []
        for text in predefined_list:
            # Extract first 3 sentences for key information
            first_sentences = extract_first_sentences(text, 3)
            processed_text = preprocess_text_for_matching(first_sentences)
            processed_list.append(processed_text if processed_text else preprocess_text_for_matching(text))
        
        # SBERT similarity on processed texts
        all_texts_sbert = processed_list + [processed_input]
        sbert_embeddings = sbert_model.encode(all_texts_sbert)
        sbert_similarities = util.cos_sim(sbert_embeddings[-1], sbert_embeddings[:-1])
        
        # TF-IDF similarity on processed texts
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts_sbert)
        tfidf_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # Calculate individual and hybrid scores
        results = []
        for i in range(len(predefined_list)):
            sbert_score = sbert_similarities[0][i].item()
            tfidf_score = tfidf_similarities[0][i]
            hybrid_score = (HYBRID_WEIGHT_SBERT * sbert_score + 
                          HYBRID_WEIGHT_TFIDF * tfidf_score)
            
            results.append({
                'index': i,
                'original_text': predefined_list[i],
                'sbert_score': sbert_score,
                'tfidf_score': tfidf_score,
                'hybrid_score': hybrid_score
            })
        
        # Sort by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        best_result = results[0]
        
        # Adaptive threshold - lower threshold for better matches
        adaptive_threshold = min(
            HYBRID_WEIGHT_SBERT * SBERT_THRESHOLD + HYBRID_WEIGHT_TFIDF * TFIDF_THRESHOLD,
            0.25  # Minimum threshold for better sensitivity
        )
        
        match_info = {
            'sbert_score': best_result['sbert_score'],
            'tfidf_score': best_result['tfidf_score'], 
            'hybrid_score': best_result['hybrid_score'],
            'hybrid_threshold': adaptive_threshold,
            'best_match': best_result['original_text'],
            'all_results': results[:5]  # Top 5 matches for comparison
        }
        
        # Enhanced evaluation logging with comparison
        eval_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": user_input,
            "processed_input": processed_input,
            "best_match": best_result['original_text'],
            "sbert_similarity": best_result['sbert_score'],
            "tfidf_similarity": best_result['tfidf_score'],
            "hybrid_similarity": best_result['hybrid_score'],
            "sbert_threshold": SBERT_THRESHOLD,
            "tfidf_threshold": TFIDF_THRESHOLD,
            "hybrid_threshold": adaptive_threshold,
            "sbert_alone_match": best_result['sbert_score'] > SBERT_THRESHOLD,
            "tfidf_alone_match": best_result['tfidf_score'] > TFIDF_THRESHOLD,
            "hybrid_match": best_result['hybrid_score'] > adaptive_threshold,
            "match_status": "success" if best_result['hybrid_score'] > adaptive_threshold else "fail"
        }
        
        if 'evaluation_log' not in st.session_state:
            st.session_state.evaluation_log = []
        st.session_state.evaluation_log.append(eval_entry)
        
        return (best_result['original_text'] if best_result['hybrid_score'] > adaptive_threshold else None, 
                match_info)
        
    except Exception as e:
        st.error(f"Error in hybrid matching: {str(e)}")
        return None, None

# --- Authentication ---
users = {"admin": "1234", "user": "abcd"}

# --- Digital Clock Component ---
def digital_clock():
    now = datetime.datetime.now()
    clock_html = f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="font-size: 24px; color: #D4AF37; font-weight: 600; margin-bottom: 10px;">
            {now.strftime("%A, %B %d, %Y")}
        </div>
        <div class="big-timer">{now.strftime("%H:%M:%S")}</div>
    </div>
    """
    st.markdown(clock_html, unsafe_allow_html=True)

# --- Login Page ---
def login():
    st.markdown("<h1 style='text-align: center'>🔐 Login</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        'logged_in': False,
        'username': "",
        'timer_start': None,
        'log_data': [],
        'active_scenario': None,
        'selected_equipment': None,
        'evaluation_log': [],
        'scenario_id': None,
        'selected_scenario_for_preview': None,
        'selected_scenario_id': None,
        'mor_data': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# --- Helper functions ---
def safe_get_value(data, key, default='N/A'):
    """Safely extract value from pandas Series or dict"""
    try:
        if isinstance(data, pd.Series):
            value = data.get(key, default)
            if isinstance(value, pd.Series):
                value = value.iloc[0] if len(value) > 0 else default
        elif isinstance(data, dict):
            value = data.get(key, default)
        else:
            value = getattr(data, key, default)
        
        # Handle NaN values
        if pd.isna(value):
            return default
        
        return str(value).strip() if str(value).strip() else default
    except Exception:
        return default

def get_scenario_id(scenario):
    """Generate a unique ID for a scenario based on its content"""
    try:
        if hasattr(scenario, 'name') and scenario.name is not None:
            return str(scenario.name)
        elif isinstance(scenario, (pd.Series, dict)):
            equipment = safe_get_value(scenario, 'Equipment', 'unknown')
            failure = safe_get_value(scenario, 'Failure Scenario', 'unknown')
            # Create a unique hash for consistent scenario IDs
            unique_string = f"{equipment}_{failure}".encode('utf-8')
            return hashlib.md5(unique_string).hexdigest()[:8]
        else:
            return f"scenario_{int(time.time() * 1000)}"
    except Exception:
        return f"scenario_{int(time.time() * 1000)}"

# --- Logout function ---
def logout():
    keys_to_reset = [
        'logged_in', 'username', 'timer_start', 'selected_equipment', 
        'active_scenario', 'scenario_id', 'selected_scenario_for_preview', 
        'selected_scenario_id'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            if key == 'logged_in':
                st.session_state[key] = False
            else:
                st.session_state[key] = None if key != 'username' else ""

# --- Check login ---
if not st.session_state.logged_in:
    login()
    st.stop()

# --- MAIN APP ---
st.title("🚄 Troubleshooting Guide System")
digital_clock()

# Sidebar
with st.sidebar:
    st.write(f"Welcome, {st.session_state.username}!")
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
    
    # Model configuration info
    st.markdown("---")
    st.subheader("Hybrid Model Info")
    st.write("Enhanced Hybrid Matching:")
    st.write(f"• SBERT Weight: {HYBRID_WEIGHT_SBERT}")
    st.write(f"• TF-IDF Weight: {HYBRID_WEIGHT_TFIDF}")
    st.write(f"• SBERT Threshold: {SBERT_THRESHOLD}")
    st.write(f"• TF-IDF Threshold: {TFIDF_THRESHOLD}")
   
    
    # Show evaluation comparison if available
    if st.session_state.get('evaluation_log'):
        # Calculate average cosine similarities
        eval_df = pd.DataFrame(st.session_state.evaluation_log)
        if not eval_df.empty:
            avg_sbert = eval_df['sbert_similarity'].mean()
            avg_tfidf = eval_df['tfidf_similarity'].mean()
            avg_hybrid = eval_df['hybrid_similarity'].mean()
            
            st.markdown("---")
            st.subheader("📈 Average Cosine Similarity")
            st.write(f"SBERT: {avg_sbert:.3f}")
            st.write(f"TF-IDF: {avg_tfidf:.3f}")
            st.write(f"Hybrid: {avg_hybrid:.3f}")
        
        if st.button("📥 Download Evaluation Data", use_container_width=True):
            csv = eval_df.to_csv(index=False)
            st.download_button(
                label="💾 Save CSV",
                data=csv,
                file_name=f"evaluation_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Download history if exists
    if os.path.exists("history.csv"):
        try:
            with open("history.csv", "rb") as f:
                st.download_button(
                    label="📥 Download History",
                    data=f,
                    file_name="failure_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error loading history file: {str(e)}")

# --- Tabs Section ---
tab1, tab2 = st.tabs(["📁 Upload MOR File", "🚄 Equipment & Scenarios"])

with tab1:
    st.header("Upload MOR File")
    st.write("Upload your equipment data file to get started.")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx'],
        help="Supported formats: CSV, XLSX (Max size: 200MB)"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.mor_data = df
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Show basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    equipment_count = df['Equipment'].nunique() if 'Equipment' in df.columns else 0
                    st.metric("Equipment Types", equipment_count)
                with col3:
                    st.metric("Columns", len(df.columns))
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

with tab2:
    if 'mor_data' not in st.session_state or st.session_state.mor_data is None:
        st.warning("⚠️ Please upload a data file first in the 'Upload MOR File' tab.")
        st.stop()

    df = st.session_state.mor_data
    
    # Validate required columns
    required_columns = ['Equipment', 'Failure Scenario']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
        st.info("Required columns: Equipment, Failure Scenario")
        st.stop()

    # --- Equipment Grid Section ---
    st.header("🔧 Select Area of Concern")
    
    try:
        equipment_list = df['Equipment'].unique().tolist()
        equipment_list = [eq for eq in equipment_list if pd.notna(eq)]  # Remove NaN values
        
        if not equipment_list:
            st.warning("No equipment found in the data.")
            st.stop()
        
        cols = st.columns(3)
        for i, equip in enumerate(equipment_list):
            with cols[i % 3]:
                if st.button(str(equip), key=f"equip_{i}", use_container_width=True):
                    st.session_state.selected_equipment = equip
                    st.session_state.active_scenario = None
                    st.session_state.scenario_id = None
                    st.rerun()
    except Exception as e:
        st.error(f"Error processing equipment list: {str(e)}")

    # --- Failure Scenarios Section ---
    if st.session_state.selected_equipment:
        st.header(f"⚠️ Failure Scenarios: {st.session_state.selected_equipment}")
        
        try:
            filtered = df[df['Equipment'] == st.session_state.selected_equipment]
            
            if filtered.empty:
                st.warning("No scenarios found for this equipment.")
            else:
                # Search functionality with hybrid model
                st.subheader("🔍 Failure Scenario Search")
                st.write("*Using enhanced hybrid SBERT + TF-IDF matching with first 3 sentences focus*")
                
                user_input = st.text_input(
                    "Enter keywords describing the scenario:",
                    placeholder="e.g., 'power failure', 'signal loss', 'mechanical fault'",
                    key="search_input"
                )
                
                # Create container for real-time evaluation results
                evaluation_container = st.container()
                
                if user_input and sbert_model:
                    predefined_scenarios = filtered['Failure Scenario'].dropna().tolist()
                    if predefined_scenarios:
                        with st.spinner("🧠 Analyzing with enhanced hybrid AI model..."):
                            match, match_info = hybrid_similarity_match(user_input, predefined_scenarios)
                        
                        # Always show evaluation results
                        with evaluation_container:
                            st.markdown("---")
                            st.subheader("📊 Real-time Cosine Similarity Evaluation")
                            
                            if match_info:
                                # Display score comparison
                                col1, col2, col3 = st.columns(3)
                                
                                # SBERT Score Visualization
                                with col1:
                                    st.markdown("### SBERT Cosine Similarity")
                                    st.markdown(f"Score: {match_info['sbert_score']:.3f}")
                                    st.markdown(f"*Range: -1 to 1*")
                                    st.markdown(f"Threshold: {SBERT_THRESHOLD:.3f}")
                                    
                                    # Visual gauge
                                    sbert_percent = min(100, max(0, (match_info['sbert_score'] + 1) * 50))
                                    st.markdown(f"""
                                        <div class="similarity-gauge">
                                            <div class="gauge-fill" style="width:{sbert_percent}%; background-color: {'#28a745' if match_info['sbert_score'] > SBERT_THRESHOLD else '#dc3545'};"></div>
                                            <div class="threshold-marker" style="left:{(SBERT_THRESHOLD + 1) * 50}%;"></div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Status indicator
                                    status = "✅ PASS" if match_info['sbert_score'] > SBERT_THRESHOLD else "❌ FAIL"
                                    st.markdown(f"<div class='{'match-success' if match_info['sbert_score'] > SBERT_THRESHOLD else 'match-fail'}'>SBERT: {status}</div>", unsafe_allow_html=True)
                                
                                # TF-IDF Score Visualization
                                with col2:
                                    st.markdown("### TF-IDF Cosine Similarity")
                                    st.markdown(f"Score: {match_info['tfidf_score']:.3f}")
                                    st.markdown(f"*Range: 0 to 1*")
                                    st.markdown(f"Threshold: {TFIDF_THRESHOLD:.3f}")
                                    
                                    # Visual gauge
                                    tfidf_percent = min(100, max(0, match_info['tfidf_score'] * 100))
                                    st.markdown(f"""
                                        <div class="similarity-gauge">
                                            <div class="gauge-fill" style="width:{tfidf_percent}%; background-color: {'#28a745' if match_info['tfidf_score'] > TFIDF_THRESHOLD else '#dc3545'};"></div>
                                            <div class="threshold-marker" style="left:{TFIDF_THRESHOLD * 100}%;"></div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Status indicator
                                    status = "✅ PASS" if match_info['tfidf_score'] > TFIDF_THRESHOLD else "❌ FAIL"
                                    st.markdown(f"<div class='{'match-success' if match_info['tfidf_score'] > TFIDF_THRESHOLD else 'match-fail'}'>TF-IDF: {status}</div>", unsafe_allow_html=True)
                                
                                # Hybrid Score Visualization
                                with col3:
                                    st.markdown("### Hybrid Score")
                                    st.markdown(f"Score: {match_info['hybrid_score']:.3f}")
                                    st.markdown(f"*Range: 0 to 1*")
                                    st.markdown(f"Threshold: {match_info['hybrid_threshold']:.3f}")
                                    
                                    # Visual gauge
                                    hybrid_percent = min(100, max(0, match_info['hybrid_score'] * 100))
                                    st.markdown(f"""
                                        <div class="similarity-gauge">
                                            <div class="gauge-fill" style="width:{hybrid_percent}%; background-color: {'#28a745' if match_info['hybrid_score'] > match_info['hybrid_threshold'] else '#dc3545'};"></div>
                                            <div class="threshold-marker" style="left:{match_info['hybrid_threshold'] * 100}%;"></div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Status indicator
                                    status = "✅ PASS" if match_info['hybrid_score'] > match_info['hybrid_threshold'] else "❌ FAIL"
                                    st.markdown(f"<div class='{'match-success' if match_info['hybrid_score'] > match_info['hybrid_threshold'] else 'match-fail'}'>HYBRID: {status}</div>", unsafe_allow_html=True)
                                
                                # Display match status
                                if match:
                                    st.markdown(f"""
                                        <div class='match-info'>
                                            <strong>✅ Perfect Match Found!</strong> 
                                            <div class='similarity-scores'>
                                                📊 SBERT Cosine: {match_info['sbert_score']:.3f} | 
                                                📊 TF-IDF Cosine: {match_info['tfidf_score']:.3f} | 
                                                📊 Hybrid Score: {match_info['hybrid_score']:.3f}
                                                (Threshold: {match_info['hybrid_threshold']:.3f})
                                            </div>
                                            <strong>Best Match:</strong> {match_info['best_match']}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Confirm match button
                                    if st.button("✔️ Confirm Match and Start Timer", type="primary"):
                                        matched_row = filtered[filtered['Failure Scenario'] == match].iloc[0]
                                        st.session_state.active_scenario = matched_row
                                        st.session_state.scenario_id = get_scenario_id(matched_row)
                                        st.session_state.timer_start = time.time()
                                        st.rerun()
                                else:
                                    st.markdown(f"""
                                        <div class='no-match-warning'>
                                            <strong>⚠️ No Exact Match Found</strong><br>
                                            Best attempt scored {match_info['hybrid_score']:.3f} (needed: {match_info['hybrid_threshold']:.3f})<br>
                                            <em>Try more specific keywords or review alternatives below</em>
                                        </div>
                                        <div style='margin-top: 10px;'>
                                            <strong>Closest Match:</strong> {match_info['best_match']}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show top 3 alternatives with similarity scores
                                    if 'all_results' in match_info and len(match_info['all_results']) > 1:
                                        st.markdown("---")
                                        st.subheader("🎯 Alternative Matches")
                                        for i, result in enumerate(match_info['all_results'][1:4], 1):
                                            score_diff = result['hybrid_score'] - match_info['hybrid_score']
                                            arrow = "↑" if score_diff > 0 else "↓"
                                            
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                st.write(f"{i}. {result['original_text'][:100]}...")
                                            with col2:
                                                st.metric("Hybrid Score", 
                                                         f"{result['hybrid_score']:.3f}", 
                                                         f"{arrow}{abs(score_diff):.3f}")
                    else:
                        st.warning("No scenarios available for matching.")

                # Available scenarios list
                st.subheader("📋 Available Scenarios")
                
                for i, (idx, row) in enumerate(filtered.iterrows()):
                    failure_scenario = safe_get_value(row, 'Failure Scenario')
                    if failure_scenario == 'N/A':
                        continue
                    
                    # Get classification and determine style
                    classification = safe_get_value(row, 'Failure Classification', 'Unknown')
                    status_class = 'status-major' if classification.lower() == 'major' else 'status-minor'
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            failure_scenario, 
                            key=f"scenario_btn_{i}",
                            use_container_width=True
                        ):
                            st.session_state.timer_start = time.time()
                            st.session_state.active_scenario = row
                            st.session_state.scenario_id = get_scenario_id(row)
                            st.rerun()
                    
                    with col2:
                        # Center the status tag vertically
                        st.markdown(
                            f"<div style='display: flex; align-items: center; height: 100%;'>"
                            f"<div class='status-tag {status_class}'>{classification}</div>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --- Scenario-Specific Information ---
if st.session_state.active_scenario is not None and st.session_state.timer_start:
    scenario = st.session_state.active_scenario
    scenario_id = st.session_state.scenario_id
    
    # Timer Display
    st.markdown("---")
    st.header("🕐 Active Scenario Timer")
    
    # Timer container with centered content
    with st.container():
        st.markdown("<div class='timer-container'>", unsafe_allow_html=True)
        
        # Display timer in center
        elapsed = time.time() - st.session_state.timer_start
        mins, secs = divmod(int(elapsed), 60)
        st.markdown(f"<div class='big-timer'>{mins:02d}:{secs:02d}</div>", unsafe_allow_html=True)
        
        # Centered Stop Timer button
        if st.button("⏹️ Stop Timer", key="stop_timer_btn", 
                     use_container_width=True, type="primary"):
            # Timer stop logic
            log_entry = {
                "Staff ID": st.session_state.username,
                "Start Time": datetime.datetime.fromtimestamp(
                    st.session_state.timer_start
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "Stop Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Date": datetime.date.today().strftime("%Y-%m-%d"),
                "Equipment": safe_get_value(scenario, 'Equipment'),
                "Failure Scenario": safe_get_value(scenario, 'Failure Scenario'),
                "Status": "Resolved" if mins < 5 else "Failed",
                "Duration (mins)": mins,
                "Guidelines": safe_get_value(scenario, 'Guidelines for the Chief Controller'),
                "Local Response": safe_get_value(scenario, 'Local Response')
            }
            
            try:
                if os.path.exists("history.csv"):
                    history_df = pd.read_csv("history.csv")
                    history_df = pd.concat([history_df, pd.DataFrame([log_entry])], ignore_index=True)
                else:
                    history_df = pd.DataFrame([log_entry])
                history_df.to_csv("history.csv", index=False)
                
                st.session_state.timer_start = None
                st.session_state.active_scenario = None
                st.session_state.scenario_id = None
                st.success("✅ Scenario logged successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving log: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close timer-container

    # Alarm Status Section
    st.markdown("---")
    st.subheader("🚨 Alarm Status")
    
    def get_alarm_status(column_name, scenario):
        """Handle both Series and scalar values properly"""
        try:
            desc = safe_get_value(scenario, column_name)
            exists = desc != 'N/A' and desc.strip() != ''
            display_text = desc if exists else 'No alarms triggered'
            return display_text, exists
        except Exception:
            return 'Column not found', False

    alarm_columns = {
        'ATS': 'ATS Alarm Description',
        'FSCADA': 'FSCADA Alarm Description', 
        'HMI': 'HMI Alarm'
    }
    
    for alarm_name, column_name in alarm_columns.items():
        display_text, exists = get_alarm_status(column_name, scenario)
        st.checkbox(
            f"{alarm_name}: {display_text}",
            value=exists,
            disabled=False,
            key=f"active_alarm_{alarm_name}_{scenario_id}"
        )

    # Response Guidelines
    st.markdown("---")
    st.subheader("📋 Guidelines for the Chief Controller")
    guidelines = safe_get_value(scenario, 'Guidelines for the Chief Controller', 'No guidelines available')
    st.markdown(f"""
        <div class='response-section'>
            <div>{guidelines}</div>
        </div>
    """, unsafe_allow_html=True)

    # Local Response Protocol
    st.markdown("---")
    st.subheader("🔧 Local Response")
    local_response = safe_get_value(scenario, 'Local Response', 'No response protocol available')
    st.markdown(f"""
        <div class='response-section'>
            <div>{local_response}</div>
        </div>
    """, unsafe_allow_html=True)

# Auto-refresh for timer
if st.session_state.timer_start:
    time.sleep(1)
    st.rerun()