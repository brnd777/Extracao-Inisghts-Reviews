# App Reviews Insights: Opinion Mining com NLP

## Sobre o Projeto

Em um cenário competitivo de aplicativos móveis, saber a nota média de um App não é suficiente para a tomada de decisão. As equipes de produto precisam saber **o porquê** da nota.

Este projeto utiliza técnicas de **Processamento de Linguagem Natural (NLP)** para minerar textos de reviews de usuários. Diferente de uma análise de sentimento tradicional (que apenas classifica entre positivo/negativo), este algoritmo identifica **quais termos específicos** são estatisticamente mais relevantes para reviews negativasXpositivas.

O objetivo é transformar texto não estruturado em **insights acionáveis** para evitar *churn* (cancelamento) e guiar o roadmap do produto.

## Tecnologias Utilizadas

* **Python** (Linguagem principal)
* **Pandas & NumPy** (Manipulação de dados)
* **Scikit-Learn** (Vetorização TF-IDF e extração de features)
* **NLTK** (Pré-processamento e Stopwords)
* **Seaborn & Matplotlib** (Visualização de dados)

## A Lógica

O diferencial deste projeto está na abordagem estatística para isolar os "drivers" de satisfação:

1.  **Limpeza Avançada:** Remoção de ruídos (links, pontuação), normalização e tratamento de stopwords personalizadas (ex: remover a palavra "app", que aparece em todos os contextos e não significa muita coisa no contexto).
2.  **Vetorização TF-IDF (N-grams):** Uso de Bigramas e Trigramas (conjuntos de duas ou três palavras juntas) (ex: "battery life", "customer support") para capturar contexto, não apenas palavras isoladas.
3.  **Cálculo Diferencial de Vetores:**
    Ao invés de apenas listar as palavras mais frequentes, apliquei a lógica:
    $$\text{Score} = \text{Vetor}_{\text{Promotores}} - \text{Vetor}_{\text{Detratores}}$$
    Isso elimina termos comuns a ambos os grupos e destaca apenas o que torna uma review **exclusivamente** boa ou ruim.

**Principais Descobertas:**
* **Pontos Fortes:** Usuários valorizam a interface ("simple easy") e que facilita a organização de hábitos e tempo ("stay organized","time management").
* **Pontos Críticos:** O termo "last update" aparece fortemente ligado a reviews negativas, indicando regressão na qualidade após a última atualização. Além disso, os termos "premium" e "create account" indicam que é necessária uma revisão nas opções premium e na criação de contas.

## Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado. Instale as dependências:

```bash
pip install -r requirements.txt
