import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import text

# --- Configurações Iniciais e Variáveis ---

# Baixar recursos necessários
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Definir as stopwords personalizadas
nltk_stop_words = set(stopwords.words('english'))
sklearn_stop_words = text.ENGLISH_STOP_WORDS
mais_palavras = ['app', 'version', 'doesnt', 'dont', 'google', 'good', 'great', 'use', 'really', 'pro', 'free','user','ive','new','recent']

final_stop_words = nltk_stop_words.union(sklearn_stop_words).union(mais_palavras)
final_stop_words = list(final_stop_words)

# Definindo funções

def clean_text_advanced(text):
    """Limpa o texto removendo links, pontuação e números."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Remove palavras comuns (stopwords) do texto."""
    return " ".join([word for word in text.split() if word not in final_stop_words])

def ler_dados_preparar(filepath):
    """Lê o CSV e faz o tratamento inicial de nulos."""
    try:
        df = pd.read_csv(filepath)
        df.dropna(subset=['content', 'score'], inplace=True)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo {filepath} não foi encontrado.")
        return None