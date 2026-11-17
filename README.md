# Manutenção Preditiva Industrial — V2.0

Aplicação demonstrativa de Ciência de Dados para estimar risco de falha a partir de variáveis operacionais de equipamentos.

**Natureza dos dados:** conjunto inteiramente sintético, criado por regras probabilísticas para fins educacionais. Os resultados não representam desempenho em máquinas reais e não devem orientar decisões operacionais.

## Funcionalidades
- Geração reproduzível de dados sintéticos de sensores.
- Análise exploratória da distribuição de falhas.
- Treinamento de Random Forest com divisão estratificada em treino, validação e teste.
- Ajuste do limiar de classificação usando apenas validação.
- Avaliação final com Precision, Recall, F1-score, ROC-AUC, Average Precision e matriz de confusão.
- Dashboard Streamlit para simular condições operacionais e consultar métricas do modelo.

## Tecnologias
Python, pandas, NumPy, scikit-learn, Matplotlib, joblib e Streamlit.

## Execução (Python 3.10+)
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/generate_data.py
python3 src/train.py
python3 -m streamlit run src/dashboard.py
```

## Estrutura
```text
├── data/
├── models/
├── reports/
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── dashboard.py
└── requirements.txt
```

## Metodologia e interpretação
A variável-alvo `failure` é gerada a partir de uma função probabilística artificial das variáveis operacionais. O classificador aprende padrões desse cenário simulado. A divisão estratificada preserva a proporção de classes nos subconjuntos.

O limiar de classificação é selecionado na validação para maximizar o F1-score da classe positiva. O conjunto de teste é reservado para a avaliação final. Em aplicações industriais reais, o limiar deve refletir custos de falsos negativos e falsos positivos, e a validação deve respeitar a ordem temporal quando houver séries históricas.

Não interprete acurácia isoladamente: em manutenção preditiva, examine especialmente Recall, Precision, F1-score, Average Precision e matriz de confusão.

## Limitações
- Dados sintéticos, sem relação com uma planta ou equipamento real.
- Variáveis e mecanismo de falha simplificados.
- Não há validação temporal, sensores físicos, integração com CLP/SCADA ou avaliação de custo operacional.
- Probabilidades não estão calibradas para uso real.

## Próximas evoluções
1. Reproduzir a análise com uma base pública documentada, como o AI4I 2020 da UCI.
2. Comparar modelos e calibrar probabilidades.
3. Aplicar validação temporal e análise de deriva.
4. Criar API e testes automatizados.
5. Documentar custos de manutenção e critérios de decisão com especialistas.
