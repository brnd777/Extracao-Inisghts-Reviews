# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

#importando funções personalizadas
from funções import ler_dados_preparar, clean_text_advanced, remove_stopwords, final_stop_words

df = ler_dados_preparar('reviews.csv')

# Tudo a partir daqui só acontece se o arquivo for carregado com sucesso
if df is not None:

    # Removendo colunas inúteis (Drop)
    colunas_para_remover = ['userImage','at','reviewCreatedVersion','repliedAt','sortOrder','replyContent','appId']
    # usando o parametro errors='ignore' evita erro se a coluna já não existir
    df.drop(columns=colunas_para_remover, axis=1, inplace=True, errors='ignore')

    # 2. Limpeza de Texto (usando funções personalizadas)
    print("Limpando reviews...")
    df['review_cleaned'] = df['content'].apply(clean_text_advanced)
    df['review_cleaned'] = df['review_cleaned'].apply(remove_stopwords)

    # 3. Vetorização

    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words=final_stop_words, min_df=3)
    X = vectorizer.fit_transform(df['review_cleaned'])

    # 4. Separando Grupos
    idx_ruins = df[df['score'] <= 2].index
    idx_boas = df[df['score'] >= 4].index

    X_ruins = X[idx_ruins].toarray()
    X_boas = X[idx_boas].toarray()

    media_ruins = X_ruins.mean(axis=0)
    media_boas = X_boas.mean(axis=0)

    # Pegando os nomes das palavras (features)
    palavras = vectorizer.get_feature_names_out()

    # Cálculo da Diferença
    top_ruins = pd.Series(media_ruins, index=palavras).sort_values(ascending=False)
    top_boas = pd.Series(media_boas, index=palavras).sort_values(ascending=False)
    
    diferenca = top_boas - top_ruins
    
    # Pontos Fortes:
    reais_pontos_fortes = diferenca.sort_values(ascending=False).head(10)

    # Pontos Fracos:
    reais_pontos_fracos = diferenca.sort_values(ascending=True).head(10).abs()

    # 5. Visualização
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Elogios
    plt.subplot(1, 2, 1)
    sns.barplot(
        x=reais_pontos_fortes.values,
        y=reais_pontos_fortes.index,
        hue=reais_pontos_fortes.index,
        palette="viridis",
        legend=False
    )
    plt.title('Destaques Positivos (Exclusivos)', fontsize=14)
    plt.xlabel('Relevância (Diferencial TF-IDF)')

    # Gráfico 2: Críticas
    plt.subplot(1, 2, 2)
    sns.barplot(
        x=reais_pontos_fracos.values,
        y=reais_pontos_fracos.index,
        hue=reais_pontos_fracos.index,
        palette="magma",
        legend=False
    )
    plt.title('Principais Reclamações (Exclusivas)', fontsize=14)
    plt.xlabel('Relevância (Diferencial TF-IDF)')

    plt.tight_layout()
    plt.show()
    