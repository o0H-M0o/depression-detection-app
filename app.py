import streamlit as st
import json
import time
import re
import requests


st.set_page_config(
    page_title="Depression Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #FFFBEB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin: 1rem 0;
    }
    .result-box-low {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .result-box-moderate {
        background-color: #FFFBEB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .result-box-high {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #DC2626;
        margin: 1rem 0;
    }
    .contact-item {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .symptom-result {
        background-color: #F8FAFC;
        padding: 0.8rem;
        border-radius: 0.4rem;
        border: 1px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .level-0 { border-left: 4px solid #10B981; }
    .level-1 { border-left: 4px solid #F59E0B; }
    .level-2 { border-left: 4px solid #EF4444; }
    .level-3 { border-left: 4px solid #7C2D12; }
</style>
""", unsafe_allow_html=True)

BDI_SYMPTOMS = {
    "symptoms": [
        {"id": "Q1", "question": "how sad the user feels"},
        {"id": "Q2", "question": "how discouraged the user is about future"},
        {"id": "Q3", "question": "how much the user feels like a failure"},
        {"id": "Q4", "question": "how much the user loses pleasure from things"},
        {"id": "Q5", "question": "how often the user feels guilty"},
        {"id": "Q6", "question": "how much the user feels punished"},
        {"id": "Q7", "question": "how much the user feels disappointed about him/herself"},
        {"id": "Q8", "question": "how often the user criticizes or blames him/herself"},
        {"id": "Q9", "question": "how much the user thinks about killing him/herself"},
        {"id": "Q10", "question": "how often the user cries"},
        {"id": "Q11", "question": "how much the user feels restless or agitated"},
        {"id": "Q12", "question": "how much the user loses interest in things"},
        {"id": "Q13", "question": "how difficult the user to make decisions"},
        {"id": "Q14", "question": "how much the user feels worthless"},
        {"id": "Q15", "question": "how much the user loses energy"},
        {"id": "Q16", "question": "how much the user experienced changes in sleeping"},
        {"id": "Q17", "question": "how much the user feels irritable"},
        {"id": "Q18", "question": "how much the user experienced changes in appetite"},
        {"id": "Q19", "question": "how difficult the user to concentrate"},
        {"id": "Q20", "question": "how much the user feels tired or fatigued"},
        {"id": "Q21", "question": "how much the user loses interest in sex"}
    ]
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # or any model you have pulled with Ollama

def ollama_generate(prompt, model=OLLAMA_MODEL):
    """Send prompt to Ollama and return the response text."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]

def build_prompt(symptom_question, user_text):
    max_chars = 4096 * 4
    truncated_text = user_text[:max_chars] + ("..." if len(user_text) > max_chars else "")
    return (
        "Suppose you are a psychiatrist who prefer to give minor diagnoses rather than serious ones, "
        f"read the posts as a whole, determine the level of \"{symptom_question}\" and give a number in 0, 1, 2 or 3, then explain why.\n"
        f"User’s posts: {user_text}\n"
        "Level (0, 1, 2 or 3):"
    )

def extract_level_and_explanation(response_text, prompt):
    """Extract the predicted level (0-3) from LLM response and return the whole response as explanation."""
    try:
        # Remove the original prompt from response
        response_only = response_text.replace(prompt, "").strip()
        
        # Try to find level using regex patterns
        level_patterns = [
            r'Level\s*\(0,\s*1,\s*2\s*or\s*3\):\s*([0-3])',   # Strict BDI format
            r'Level:\s*([0-3])',                              # "Level: 2"
            r'^([0-3])',                                      # Level at start
            r'([0-3])\s*[-\.]',                               # Level followed by dash or dot
            r'Level.*?([0-3])',                               # Any "Level..." with a digit
            r'([0-3])\s*(?:out of|/)\s*3',                    # "2 out of 3" or "1/3"
            r'(?:answer|score|level|rating).*?([0-3])',       # "score is 2", etc.
            r'([0-3])(?=\s|$|\.)',   
        ]
        
        level = None
        for pattern in level_patterns:
            match = re.search(pattern, response_only, re.MULTILINE | re.IGNORECASE)
            if match:
                level = int(match.group(1))
                break
        
        if level is None:
            # Fallback: look for any digit 0-3
            digit_match = re.search(r'[0-3]', response_only)
            level = int(digit_match.group()) if digit_match else 0
        
        # For explanation, just return the whole response (truncated)
        explanation = response_only if response_only else "Analysis completed."
        
        return level, explanation
        
    except Exception as e:
        return 0, f"Error parsing response: {str(e)}"

def analyze_symptom(symptom_question, user_text):
    try:
        prompt = build_prompt(symptom_question, user_text)
        response_text = ollama_generate(prompt)
        level, explanation = extract_level_and_explanation(response_text, prompt)
        return level, explanation
    except Exception as e:
        return 0, f"Error analyzing symptom: {str(e)}"

def get_level_color_class(level):
    """Get CSS class based on level"""
    return f"level-{level}"

def get_severity_text(level):
    """Convert numeric level to text"""
    severity_map = {0: "Minimal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    return severity_map.get(level, "Unknown")

def calculate_total_score(results):
    """Calculate total BDI score"""
    return sum(result['level'] for result in results)

def interpret_total_score(total_score):
    """Interpret total BDI score"""
    if total_score <= 9:
        return "Minimal", "result-box-low"
    elif total_score <= 19:
        return "Mild", "result-box-moderate"
    elif total_score <= 29:
        return "Moderate", "result-box-moderate"
    else:
        return "Severe", "result-box-high"

# --- Navigation ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏡 Home"

st.sidebar.image("image/emotions.jpg")
if st.sidebar.button("🏡 Home", use_container_width=True):
    st.session_state.current_page = "🏡 Home"
if st.sidebar.button("🕵️ Depression Detection", use_container_width=True):
    st.session_state.current_page = "🕵️ Depression Detection"
if st.sidebar.button("📬 Contact", use_container_width=True):
    st.session_state.current_page = "📬 Contact"

st.sidebar.markdown("---")

# --- Page Content ---
if st.session_state.current_page == "🏡 Home":
    st.markdown('<div class="main-header">🧠 Depression Detection System</div>', unsafe_allow_html=True)
   
    st.image("image/emotions.jpg")
    st.markdown('<div class="sub-header">About the System</div>', unsafe_allow_html=True)
    st.markdown("""
    This system use large language models to analyze text input 
    and estimate the severity of depression symptoms and risk of depression.
    """)
    st.markdown('<div class="disclaimer">❗️REMINDER: This is not a medical diagnosis</div>', unsafe_allow_html=True)

    st.markdown("""
                This analysis is based on text patterns only and should not replace professional medical advice. 
                If you are concerned about your mental health, please consult with a healthcare professional.
                """)

elif st.session_state.current_page == "🕵️ Depression Detection":
   

    st.markdown('<div class="main-header">🕵️ Depression Detection System</div>', unsafe_allow_html=True)
    st.markdown("""
    Enter some text below (like a social media post, diary sentences, or any written thoughts) and click "Analyze" to receive an assessment.
    """)
    consent = st.checkbox("I consent to the analysis of my text for mental health assessment. I understand that my data will not be stored or shared.")
    
    # Add this above your text input in the Streamlit app

    with st.expander("💡 Tips for writing your text (click to expand)"):
        st.markdown("""
        To get more accurate analysis, try to mention how you have been feeling and any changes you’ve noticed.
        Examples of areas to include:
        - **Feelings:** Describe your emotional state, such as feeling sad, discouraged about the future, or like a failure.
        - **Enjoyment:** Talk about whether you still find pleasure in things you used to enjoy, or if you’ve lost interest in activities.
        - **Self-perception:** Share if you’ve been feeling guilty, worthless, disappointed in yourself, or overly self-critical.
        - **Thoughts:** Mention any hopeless or negative thoughts about the future, or if you’ve had thoughts of self-harm.
        - **Emotional reactions:** Note if you’ve been crying more often, feeling restless, agitated, or irritable.
        - **Daily life:** Describe any changes in your sleep patterns, appetite, energy levels, or if you often feel tired or fatigued.
        - **Cognition:** Include if you’ve had trouble concentrating, making decisions, or if you feel mentally slowed down.
        - **Relationships and interest:** Mention if you’ve lost interest in sex or social activities, or if you feel disconnected from others.

        Just share what feels most relevant to you.
        """)
    
  
    user_text = st.text_area("Enter text to analyze", height=200, placeholder="Type or paste your text here...")
    if st.button("Analyze Text"):
        if not consent:
            st.warning("Please check the box to consent to the analysis before proceeding.")
            st.stop()
        if user_text.strip() == "":
            st.error("Please enter some text for analysis.")
            st.stop()
          
        st.markdown("""
            **Disclaimer:** This analysis is based on text patterns only and should not replace professional medical advice. 
            If you are concerned about your mental health, please consult with a healthcare professional.
            """)
        
        # Create containers for real-time updates
        progress_bar = st.progress(0)
        current_symptom_container = st.empty()
        results_container = st.container()
        
        # Initialize results storage
        analysis_results = []
        symptoms = BDI_SYMPTOMS["symptoms"]

        # Analyze each symptom
        for i, symptom in enumerate(symptoms):
            # Update progress
            progress = (i + 1) / len(symptoms)
            progress_bar.progress(progress)
            current_symptom_container.info(f"🔄 Analyzing: {symptom['question']} ({i+1}/{len(symptoms)})")
            
            # Analyze symptom
            level, explanation = analyze_symptom(symptom['question'], user_text)
            
            # Store result
            result = {
                'id': symptom['id'],
                'question': symptom['question'],
                'level': level,
                'explanation': explanation
            }
            analysis_results.append(result)
            
            # Display result immediately
            with results_container:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{symptom['id']}: {symptom['question'].title()}**")
                with col2:
                    severity = get_severity_text(level)
                    color_class = get_level_color_class(level)
                    st.markdown(f'<div class="symptom-result {color_class}"><strong>Level {level} - {severity}</strong></div>', unsafe_allow_html=True)
                
                with st.expander(f"Explanation for {symptom['id']}", expanded=False):
                    st.write(explanation)
                
                st.markdown("---")
        
        # Clear progress indicators
        progress_bar.empty()
        current_symptom_container.empty()
        
        # Calculate and display overall results
        st.markdown('<div class="sub-header">📊 Overall Assessment</div>', unsafe_allow_html=True)
        
        total_score = calculate_total_score(analysis_results)
        severity_level, severity_class = interpret_total_score(total_score)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total BDI Score", total_score, help="Sum of all symptom levels (0-63)")
        with col2:
            st.metric("Severity Level", severity_level)
        with col3:
            high_symptoms = len([r for r in analysis_results if r['level'] >= 2])
            st.metric("High-Level Symptoms", high_symptoms)
        
        st.markdown(f'<div class="{severity_class}"><strong>Overall Assessment: {severity_level} Depression Indicators</strong></div>', unsafe_allow_html=True)
        
        # Interpretation guide
        st.markdown("### Score Interpretation")
        st.markdown("""
        - **0-9**: Minimal depression
        - **10-19**: Mild depression  
        - **20-29**: Moderate depression
        - **29-63**: Severe depression
        """)
        
        # Summary of concerning symptoms
        concerning_symptoms = [r for r in analysis_results if r['level'] >= 2]
        if concerning_symptoms:
            st.markdown("### Symptoms Requiring Attention")
            for symptom in concerning_symptoms:
                st.warning(f"**{symptom['id']}**: {symptom['question']} - Level {symptom['level']}")
        
        st.markdown('<div class="disclaimer">**Remember**: This analysis is based on text patterns and should not be used for self-diagnosis. Please consult a mental health professional for proper evaluation and treatment.</div>', unsafe_allow_html=True)


elif st.session_state.current_page == "📬 Contact":
    st.markdown('<div class="main-header">📬 Contact</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Song Hui Min</div>', unsafe_allow_html=True)
    st.markdown('<div class="contact-item"><strong>Bachelor of Computer Science (Information System), Faculty of Computer Science and Information Technology, University of Malaya</div>', unsafe_allow_html=True)
    st.markdown('<div class="contact-item"><strong>Email:</strong> 22053612@siswa.um.edu.my</div>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div class="sub-header">Associate Prof Dr. Kasturi Dewi A/P Varathan</div>', unsafe_allow_html=True)
    st.markdown('<div class="contact-item"><strong>Department of Information System, Faculty of Computer Science and Information Technology, University of Malaya</div>', unsafe_allow_html=True)
    st.markdown('<div class="contact-item"><strong>Email:</strong> kasturi@um.edu.my</div>', unsafe_allow_html=True)
