"""
Titanic Survival Predictor
Aplikacja Streamlit do predykcji przeżycia pasażera Titanica.
Model: regresja logistyczna wytrenowana w Google Colab.

Autor: Natalia Traczewska
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from pathlib import Path
from datetime import datetime

# =====================================================================
# KONFIGURACJA LOGGINGU
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# KONFIGURACJA STRONY
# =====================================================================
st.set_page_config(
    page_title='Czy przeżyłbyś katastrofę "Titanica"?',
    page_icon='🚢',
    layout='centered',
    initial_sidebar_state='expanded',
)

# =====================================================================
# STYL CSS
# =====================================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.8em;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2em;
        margin-top: 0;
    }
    .survived-card {
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    .died-card {
        background: linear-gradient(135deg, #34495E, #2C3E50);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
    }
    .confidence-high {
        color: #27AE60;
        font-weight: bold;
    }
    .confidence-low {
        color: #E74C3C;
        font-weight: bold;
    }
    .info-box {
        background: #E8F4F8;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# SŁOWNIKI MAPUJĄCE
# =====================================================================
sex_d = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d = {1: 'Pierwsza', 2: 'Druga', 3: 'Trzecia'}
embarked_d = {0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'}

# =====================================================================
# WCZYTANIE MODELU
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Wczytuje wytrenowany model z pliku pickle.
    
    Returns:
        dict: Słownik zawierający model, scaler i metadane
        
    Raises:
        FileNotFoundError: Gdy plik modelu nie istnieje
        Exception: Gdy błąd podczas wczytywania pliku
    """
    model_path = 'model.pickle'
    
    # Sprawdzenie istnienia pliku
    if not os.path.exists(model_path):
        logger.error(f'Model file not found: {model_path}')
        raise FileNotFoundError(
            f'Nie znaleziono pliku modelu: {model_path}\n'
        )
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info('Model loaded successfully')
        return model_data
    except pickle.UnpicklingError as e:
        logger.error(f'Pickle error: {e}')
        raise Exception(f'Błąd wczytywania modelu: Uszkodzony plik pickle.\n{str(e)}')
    except Exception as e:
        logger.error(f'Unexpected error loading model: {e}')
        raise Exception(f'Błąd wczytywania modelu: {str(e)}')



# =====================================================================
# FUNKCJE POMOCNICZE
# =====================================================================
def validate_passenger_data(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Waliduje dane pasażera pod kątem logicznych błędów.
    
    Args:
        pclass, sex, age, sibsp, parch, fare, embarked: Parametry pasażera
        
    Returns:
        tuple: (bool, str) - (czy dane są poprawne, komunikat błędu jeśli jest)
    """
    warnings = []
    
    # Walidacja klasy
    if pclass not in [1, 2, 3]:
        return False, 'Błąd: Nieprawidłowa klasa kajuty'
    
    # Walidacja płci
    if sex not in [0, 1]:
        return False, 'Błąd: Nieprawidłowa płeć'
    
    # Walidacja wieku
    if age < 0 or age > 120:
        return False, 'Błąd: Wiek musi być między 0 a 120 lat'
    
    if age == 0 and (sibsp > 0 or parch == 0):
        warnings.append('Noworodek bez rodziców na pokładzie?')
    
    # Walidacja rodziny
    if sibsp < 0 or parch < 0:
        return False, 'Błąd: Liczba osób z rodziny nie może być ujemna'
    
    if sibsp + parch > 10:
        warnings.append('Bardzo duża rodzina - warte uwagi!')
    
    # Walidacja opłaty
    if fare < 0:
        return False, 'Błąd: Opłata za bilet nie może być ujemna'
    
    if fare == 0 and pclass > 1:
        warnings.append('Bilet za darmo w drugiej/trzeciej klasie?')
    
    # Walidacja portu
    if embarked not in [0, 1, 2]:
        return False, 'Błąd: Nieprawidłowy port wysiadki'
    
    # Zwrócenie wyniku
    if warnings:
        return True, ' | '.join(warnings)
    return True, ''

def create_prediction_summary(pclass, sex, age, sibsp, parch, fare, embarked, 
                              prob_died, prob_survived, prediction):
    """Tworzy szczegółowe podsumowanie predykcji."""
    confidence = max(prob_died, prob_survived)
    is_confident = confidence > 70
    
    return {
        'confident': is_confident,
        'confidence': confidence,
        'prediction': 'PRZEŻYŁBYŚ!' if prediction == 1 else 'NIE PRZEŻYŁBYŚ...',
        'survived_prob': prob_survived,
        'died_prob': prob_died,
    }

with st.sidebar:
    st.header('O aplikacji')
    st.write(
        '**Titanic Survival Predictor** to aplikacja wykorzystująca '
        'model **regresji logistycznej** do przewidywania szans '
        'przeżycia pasażera katastrofy Titanic.'
    )

    st.subheader('Instrukcja')
    st.markdown(
        """
        1. **Ustaw parametry pasażera** za pomocą widgetów
        2. **Kliknij** „Przewiduj przeżycie"
        3. **Zobacz wynik** wraz z prawdopodobieństwem
        """
    )

    st.subheader('O modelu')
    st.markdown(
        """
        - **Algorytm**: Regresja logistyczna
        - **Biblioteka**: scikit-learn
        - **Cechy** (7): klasa, płeć, wiek, rodzeństwo, rodzice, opłata, port
        - **Dataset**: Titanic (seaborn)
        """
    )

    st.markdown('---')
    st.caption('Lab06 – Środowiska AutoML')

# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    '<h1 class="main-title">Titanic Survival Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Czy przeżyłbyś katastrofę "Titanica"?</p>',
    unsafe_allow_html=True,
)

# Grafika Titanica
st.image(
    'https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG',
    use_container_width=True,
    caption='RMS Titanic, kwiecień 1912',
)

st.markdown('---')

# =====================================================================
# WIDGETY – DANE PASAŻERA
# =====================================================================
st.subheader('Dane pasażera')

# Wiersz 1: Klasa + Płeć
col1, col2 = st.columns(2)

with col1:
    pclass = st.radio(
        '**Klasa kajuty:**',
        list(pclass_d.keys()),
        format_func=lambda x: pclass_d[x],
    )

with col2:
    sex = st.radio(
        '**Płeć:**',
        list(sex_d.keys()),
        format_func=lambda x: sex_d[x],
    )

# Wiersz 2: Wiek (suwak)
age = st.slider(
    '**Wiek:**',
    min_value=0,
    max_value=80,
    value=30,
    step=1,
    help='Wiek pasażera w latach',
)

# Wiersz 3: Rodzina
col3, col4 = st.columns(2)

with col3:
    sibsp = st.number_input(
        '**Liczba rodzeństwa/małżonka:**',
        min_value=0,
        max_value=8,
        value=0,
        help='Bracia, siostry, mąż lub żona na pokładzie',
    )

with col4:
    parch = st.number_input(
        '**Liczba rodziców/dzieci:**',
        min_value=0,
        max_value=6,
        value=0,
        help='Rodzice lub dzieci na pokładzie',
    )

# Wiersz 4: Opłata + Port
col5, col6 = st.columns(2)

with col5:
    fare = st.slider(
        '**Opłata za bilet (£):**',
        min_value=0.0,
        max_value=520.0,
        value=32.0,
        step=0.5,
        help='Cena biletu w funtach',
    )

with col6:
    embarked = st.selectbox(
        '**Port wejścia na pokład:**',
        list(embarked_d.keys()),
        format_func=lambda x: embarked_d[x],
    )

st.markdown('---')

# =====================================================================
# PRZYCISK PREDYKCJI
# =====================================================================
predict_btn = st.button(
    'Przewiduj przeżycie',
    type='primary',
    use_container_width=True,
)

# =====================================================================
# LOGIKA PREDYKCJI
# =====================================================================
if predict_btn:
    try:
        # Walidacja danych wejściowych
        is_valid, validation_msg = validate_passenger_data(
            pclass, sex, age, sibsp, parch, fare, embarked
        )
        
        if not is_valid:
            st.error(validation_msg)
            logger.warning(f'Invalid input: {validation_msg}')
        else:
            if validation_msg:
                st.warning(validation_msg)
            
            # Wczytanie modelu
            with st.spinner('Ładuję model...'):
                try:
                    model_data = load_model()
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_names = model_data.get('feature_names', [])
                except FileNotFoundError as e:
                    st.error(str(e))
                    logger.error(f'Model file not found: {e}')
                    st.stop()
                except Exception as e:
                    st.error(str(e))
                    logger.error(f'Error loading model: {e}')
                    st.stop()

            # Przygotowanie danych wejściowych (kolejność: pclass, sex, age, sibsp, parch, fare, embarked)
            try:
                input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
                
                # Sprawdzenie poprawności kształtu danych
                if input_data.shape[1] != 7:
                    raise ValueError(f'Oczekiwano 7 cech, otrzymano {input_data.shape[1]}')
                
                input_scaled = scaler.transform(input_data)
                logger.info(f'Data prepared: {input_data}')
            except Exception as e:
                st.error(f'Błąd przygotowania danych: {str(e)}')
                logger.error(f'Data preparation error: {e}')
                st.stop()

            # Predykcja
            try:
                with st.spinner('🔬 Analizuję szanse przeżycia...'):
                    prediction = model.predict(input_scaled)[0]
                    probabilities = model.predict_proba(input_scaled)[0]

                prob_died = probabilities[0] * 100
                prob_survived = probabilities[1] * 100
                
                logger.info(f'Prediction made: survived={prediction}, probs=[{prob_died:.2f}%, {prob_survived:.2f}%]')
                
            except Exception as e:
                st.error(f'Błąd podczas predykcji: {str(e)}')
                logger.error(f'Prediction error: {e}')
                st.stop()

            # Tworz podsumowanie
            summary = create_prediction_summary(
                pclass, sex, age, sibsp, parch, fare, embarked,
                prob_died, prob_survived, prediction
            )

            # ----------- WYNIK GŁÓWNY -----------
            if prediction == 1:
                st.balloons()
                confidence_indicator = 'PEWNA PREDYKCJA' if summary['confident'] else 'NISKA PEWNOŚĆ'
                st.markdown(
                    f"""
                    <div class="survived-card">
                        <h1>PRZEŻYŁBYŚ!</h1>
                        <h2>Szansa przeżycia: {prob_survived:.1f}%</h2>
                        <p style="margin-top: 10px; font-size: 0.9em;">{confidence_indicator} ({summary['confidence']:.1f}%)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                confidence_indicator = '✅ PEWNA PREDYKCJA' if summary['confident'] else 'NISKA PEWNOŚĆ'
                st.markdown(
                    f"""
                    <div class="died-card">
                        <h1>NIE PRZEŻYŁBYŚ...</h1>
                        <h2>Szansa przeżycia: {prob_survived:.1f}%</h2>
                        <p style="margin-top: 10px; font-size: 0.9em;">{confidence_indicator} ({summary['confidence']:.1f}%)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------- WYKRES PRAWDOPODOBIEŃSTW -----------
        st.subheader('Szczegółowe prawdopodobieństwa')

        try:
            df_proba = pd.DataFrame({
                'Wynik': ['Nie przeżył', 'Przeżył'],
                'Prawdopodobieństwo (%)': [prob_died, prob_survived],
                'Kolor': ['#34495E', '#2ECC71'],
            })

            fig = px.bar(
                df_proba,
                x='Prawdopodobieństwo (%)',
                y='Wynik',
                orientation='h',
                text=df_proba['Prawdopodobieństwo (%)'].map(lambda v: f'{v:.1f}%'),
                color='Wynik',
                color_discrete_map={
                    'Nie przeżył': '#34495E',
                    'Przeżył': '#2ECC71',
                },
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title='Prawdopodobieństwo (%)',
                yaxis_title='',
                xaxis_range=[0, 100],
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig.update_traces(textposition='outside', textfont_size=14)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f'Błąd rysowania wykresu: {str(e)}')
            logger.error(f'Plot error: {e}')

        # ----------- PODSUMOWANIE PASAŻERA -----------
        with st.expander('Twoje dane pasażera'):
            try:
                summary_data = pd.DataFrame({
                    'Cecha': [
                        'Klasa kajuty',
                        'Płeć',
                        'Wiek',
                        'Rodzeństwo/małżonek',
                        'Rodzice/dzieci',
                        'Opłata za bilet',
                        'Port wejścia',
                    ],
                    'Wartość': [
                        pclass_d.get(pclass, 'Nieznana'),
                        sex_d.get(sex, 'Nieznana'),
                        f'{age} lat',
                        f'{int(sibsp)} os.',
                        f'{int(parch)} os.',
                        f'£{fare:.2f}',
                        embarked_d.get(embarked, 'Nieznany'),
                    ],
                })
                st.dataframe(summary_data, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f'Błąd wyświetlania podsumowania: {str(e)}')
                logger.error(f'Summary display error: {e}')

        # ----------- INTERPRETACJA -----------
        st.subheader('Co wpłynęło na wynik?')

        try:
            factors = []
            if sex == 0:
                factors.append('**Kobieta** — w 1912 obowiązywała zasada „kobiety i dzieci najpierw"')
            else:
                factors.append('**Mężczyzna** — szanse przeżycia były znacznie mniejsze')

            if pclass == 1:
                factors.append('**Pierwsza klasa** — bliżej szalup ratunkowych')
            elif pclass == 3:
                factors.append('**Trzecia klasa** — pasażerowie byli daleko od pokładu')

            if age < 16:
                factors.append('**Dziecko** — dzieci miały pierwszeństwo')
            elif age > 60:
                factors.append('**Starszy wiek** — utrudniał ewakuację')

            if fare > 100:
                factors.append('**Wysoka opłata** — sugeruje lepszą lokalizację kajuty')

            if sibsp + parch == 0:
                factors.append('**Samotny pasażer** — nikt nie wymagał ratunku poza Tobą')
            elif sibsp + parch >= 4:
                factors.append('**Duża rodzina** — trudno ewakuować się razem')

            if not factors:
                st.info('Brak istotnych czynników dla tego profilu.')
            else:
                for factor in factors:
                    st.markdown(f'- {factor}')

            # Dokładność modelu
            if model_data and 'accuracy' in model_data:
                accuracy = model_data.get('accuracy', 0)
                st.divider()
                if accuracy > 0:
                    st.caption(f'Dokładność modelu na zbiorze testowym: **{accuracy*100:.1f}%**')
                    
        except Exception as e:
            st.error(f'Błąd interpretacji wyników: {str(e)}')
            logger.error(f'Interpretation error: {e}')
        
        # Dodatkowa informacja o modelu
        st.info(
            '**Informacja**: Ta predykcja jest oparta na modelu regresji logistycznej '
            'wytrenowanym na rzeczywistych danych z katastrofy Titanica. '
            'Nie jest 100% dokładna, ale pokazuje statystyczne trendy z 1912 roku.'
        )
        
    except Exception as e:
        st.error(f'Nieoczekiwany błąd w aplikacji: {str(e)}')
        logger.error(f'Unexpected application error: {e}', exc_info=True)
        st.info('Spróbuj przeładować stronę.')

# =====================================================================
# STOPKA
# =====================================================================
st.markdown('---')
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <b>Lab06 – Środowiska uruchomieniowe AutoML</b><br>
        Autorka: <b>Natalia Traczewska</b>
    </div>
    """,
    unsafe_allow_html=True,
)