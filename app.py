import streamlit as st
import os
import pandas as pd
import boto3
import joblib
from dotenv import load_dotenv
import json
import re
from langfuse import Langfuse
# from langfuse import Langfuse
from openai import OpenAI
import uuid

load_dotenv()

# --- TESTOWY TRACE LANGFUSE ---
#try:
#    
#    test_client = Langfuse(
#        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
#        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
#        host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
#    )
#    test_trace = test_client.trace(name="test_trace_startup", input={"msg": "test trace from app1.py"})
#    test_trace.end(output={"msg": "test trace zakończony"})
#    test_client.flush()
#    print("✅ Testowy trace Langfuse wysłany!")
#except Exception as e:
#    print(f"❌ Testowy trace Langfuse NIE wysłany: {e}")

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Pobierz model z Digital Ocean Spaces jeśli nie istnieje lokalnie
MODEL_PATH = 'best_model_pipeline.joblib'
BUCKET_NAME = 'kursai'

@st.cache_resource
def load_model():
    """Załaduj model z cache'owaniem"""
    if not os.path.exists(MODEL_PATH):
        session = boto3.session.Session()
        client = session.client('s3')
        client.download_file(BUCKET_NAME, MODEL_PATH, MODEL_PATH)
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_langfuse_client():
    """Inicjalizuj klienta Langfuse z cache'owaniem"""
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=os.environ('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.environ('LANGFUSE_SECRET_KEY'),
            host=os.environ('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
        # Test connection
        client.auth_check()
        return client
    except Exception as e:
        print(f"Langfuse initialization failed: {e}")
        return None

def parse_pace_to_seconds(pace_str):
    """Konwertuj tempo z różnych formatów na sekundy na km"""
    if not pace_str:
        return None
    
    # Usuń białe znaki i zmień na lowercase
    pace_str = str(pace_str).strip().lower()
    
    # Wzorce do rozpoznawania tempa
    patterns = [
        r'(\d+):(\d+)/km',  # 5:10/km
        r'(\d+):(\d+)',     # 5:10
        r'(\d+)\.(\d+)',    # 5.10
        r'(\d+),(\d+)',     # 5,10
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pace_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
    
    # Sprawdź czy to już sekundy
    try:
        seconds = float(pace_str.replace('/km', '').replace('s', ''))
        if 200 <= seconds <= 600:  # Realistyczne tempo w sekundach (3:20-10:00/km)
            return int(seconds)
    except:
        pass
    
    return None

def extract_features_with_regex(text):
    """Wyodrębnij cechy używając regex jako backup"""
    features = {}
    
    # Płeć
    if any(word in text.lower() for word in ['kobieta', 'kobietą', 'female', 'woman']):
        features['gender'] = 'kobieta'
    elif any(word in text.lower() for word in ['mężczyzna', 'mężczyzną', 'male', 'man', 'chłopak']):
        features['gender'] = 'mężczyzna'
    
    # Wiek
    age_match = re.search(r'(\d+)\s*(?:lat|roku|years?|old)', text.lower())
    if age_match:
        features['age'] = int(age_match.group(1))
    
    # Tempo
    pace_patterns = [
        r'tempo.*?(\d+):(\d+)',
        r'(\d+):(\d+)/km',
        r'(\d+):(\d+)\s*na\s*km',
        r'(\d+):(\d+)\s*min',
    ]
    
    for pattern in pace_patterns:
        match = re.search(pattern, text.lower())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            features['pace_5k'] = minutes * 60 + seconds
            break
    
    return features



# Inicjalizacja Langfuse (raz globalnie)
langfuse = Langfuse(
    public_key=os.environ('LANGFUSE_PUBLIC_KEY'),
    secret_key=os.environ('LANGFUSE_SECRET_KEY'),
    host=os.environ('LANGFUSE_HOST', 'https://cloud.langfuse.com')
)
langfuse.auth_check()

llm_client = OpenAI(api_key=os.environ("OPENAI_API_KEY"))

def extract_features_from_text(text, openai_api_key):
    # Prompt do OpenAI
    prompt = f"""Przeanalizuj poniższy tekst i wyodrębnij informacje w formacie JSON.

WYMAGANY FORMAT ODPOWIEDZI:
{{
  "gender": "kobieta" lub "mężczyzna",
  "age": liczba_jako_integer,
  "pace_5k": liczba_sekund_na_kilometr_jako_integer
}}

INSTRUKCJE:
1. Dla płci: znajdź słowa jak "kobieta", "kobietą", "mężczyzna", "mężczyzną"
2. Dla wieku: znajdź liczbę przed słowami "lat", "lata", "roku"
3. Dla tempa: znajdź format MM:SS i zamień na sekundy (np. 5:10 = 310 sekund)

TEKST: "{text}"

Odpowiedz TYLKO kodem JSON, bez żadnych innych słów:"""

    messages = [
        {"role": "system", "content": "Jesteś ekspertem w wyodrębnianiu danych. Odpowiadasz TYLKO kodem JSON."},
        {"role": "user", "content": prompt}
    ]

    # --- Langfuse trace ---
    trace = langfuse.trace(
        name="extract_features_from_text",
        input=messages,
        metadata={"app": "streamlit", "model": "gpt-3.5-turbo"}
    )
    span = trace.span(name="openai-extract", input={"text": text})

    # --- OpenAI call ---
    client = OpenAI(api_key=openai_api_key)
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=100,
        response_format={"type": "json_object"}
    )
    resp = chat_completion.choices[0].message.content

    # --- Langfuse generation ---
    generation = span.generation(
        name="data-extraction",
        model="gpt-3.5-turbo",
        input=messages
    )
    generation.end(
        output=chat_completion.choices,
        usage={
            "input": chat_completion.usage.prompt_tokens,
            "output": chat_completion.usage.completion_tokens,
            "total": chat_completion.usage.total_tokens,
            "unit": "TOKENS"
        }
    )
    span.end(output=resp)

    # --- Parsowanie odpowiedzi ---
    try:
        output = json.loads(resp)
    except Exception:
        output = {"error": resp}

    trace.update(output=output)
    langfuse.flush()

    return output, resp

def validate_features(features):
    """Sprawdź czy wszystkie wymagane cechy są obecne"""
    missing = []
    if not features.get('gender'):
        missing.append('płeć')
    if not features.get('age'):
        missing.append('wiek')
    if not features.get('pace_5k'):
        missing.append('tempo na 5km')
    return missing

def format_time(seconds):
    """Zamień sekundy na format hh:mm:ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def prepare_prediction_data(features):
    """Przygotuj dane do predykcji"""
    input_dict = {
        'gender': 1 if features['gender'].lower() in ['kobieta', 'female'] else 0,
        'age': int(features['age']),
        'pace_5k': float(features['pace_5k'])
    }
    
    # Konwersja sekund na minuty jeśli potrzeba
    if input_dict['pace_5k'] > 20:
        input_dict['pace_5k'] = input_dict['pace_5k'] / 60
    
    return pd.DataFrame([input_dict])

def main():
    st.title("🏃‍♀️ Predykcja czasu półmaratonu")
    st.write("Podaj swoje dane w naturalnym języku")
    
    # Przykłady
    with st.expander("💡 Zobacz przykłady"):
        st.write("• Mam na imię Kasia, mam 32 lata, jestem kobietą, moje tempo na 5km to 5:10/km")
        st.write("• Jestem Piotrem, 28 lat, mężczyzna, biegam 5km w tempie 4:30/km")
        st.write("• Anna, 25 lat, kobieta, tempo 5:45")
    
    # Sprawdź klucze API
    openai_api_key = os.environ('OPENAI_API_KEY')
    
    if not openai_api_key:
        st.error("❌ Brak OPENAI_API_KEY w zmiennych środowiskowych")
        return
    
    # Test Langfuse (opcjonalne)
    langfuse_status = get_langfuse_client()
    if langfuse_status:
        try:
            import langfuse
            version = langfuse.__version__
            st.success(f"✅ Langfuse połączony (v{version})")
        except:
            st.success("✅ Langfuse połączony")
    else:
        st.warning("⚠️ Langfuse niedostępny (aplikacja będzie działać bez logowania)")
    
    # Załaduj model
    try:
        model = load_model()
        st.success("✅ Model załadowany pomyślnie")
    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu: {e}")
        return
    
    user_input = st.text_area(
        "Opisz siebie:", 
        placeholder="Przykład: Jestem Anną, mam 28 lat, jestem kobietą i moje tempo na 5km to 4:50/km",
        height=100
    )
    
    if st.button("🎯 Przewiduj czas półmaratonu", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Proszę wprowadzić opis")
            return
            
        with st.spinner('🤖 Analizuję dane...'):
            # Wyodrębnij cechy
            features, llm_response = extract_features_from_text(user_input, openai_api_key)
            
            # Debug info dla użytkownika
            #st.write("🔍 **Wyodrębnione dane:**")
            #st.json(features)
            
            # Sprawdź completność danych
            missing = validate_features(features)
            
            if missing:
                st.error(f"❌ Brakuje następujących danych: {', '.join(missing)}")
                
                with st.expander("🔍 Zobacz surową odpowiedź"):
                    st.code(llm_response)
                
                st.info("💡 Spróbuj być bardziej precyzyjny, np.: 'Jestem kobietą, mam 30 lat, tempo na 5km to 5:15/km'")
            else:
                try:
                    # Przygotuj dane i wykonaj predykcję
                    X = prepare_prediction_data(features)
                    prediction = model.predict(X)[0]
                    
                    # Wyświetl wyniki
                    st.success(f"🎯 **Przewidywany czas półmaratonu: {format_time(prediction)}**")
                    
                    st.write("🔍 Dla parametrów:")
                    # Dodatkowe informacje
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Płeć", features['gender'])
                    with col2:
                        st.metric("Wiek", f"{features['age']} lat")
                    with col3:
                        pace_min = features['pace_5k'] // 60
                        pace_sec = features['pace_5k'] % 60
                        st.metric("Tempo 5km", f"{pace_min}:{pace_sec:02d}/km")
                    
                    # Sprawdź czy czas jest realistyczny
                    if prediction < 3000 or prediction > 15000:
                        st.warning("⚠️ Model przewidział nietypowy czas. Sprawdź poprawność danych!")
                        
                except Exception as e:
                    st.error(f"❌ Błąd podczas predykcji: {e}")
                    st.write("Debug info:", features)

if __name__ == "__main__":
    main()