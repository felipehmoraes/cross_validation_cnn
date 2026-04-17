📌 Descrição

Este projeto implementa validação cruzada (cross-validation) em Python.
Por padrão, utiliza 5 folds, ou seja, o treinamento é executado cinco vezes sobre o dataset.

⚙️ Execução

O projeto possui duas formas de execução:

Automatizada (.bat): permite testar múltiplas resoluções de forma automática com apenas um clique.
Execução direta: realiza o treinamento com uma resolução fixa.
🧠 Configuração
Número de folds: 5
Épocas: 30
Framework: TensorFlow/Keras
Uso de GPU com CUDA
💻 Hardware utilizado
GPU utilizada nos testes: NVIDIA RTX 3050 Mobile
📊 Saídas geradas por fold

Ao final de cada fold, o projeto gera:

Um gráfico de acurácia
Um gráfico de loss
Um gráfico de precisão
Um gráfico de recall
Uma matriz de confusão por fold
Uma matriz de confusão com os nomes dos modelos ocultos por fold
Um arquivo de treino/modelo no formato .h5 por fold
📁 Arquivos gerados na pasta de resultados

Dentro da pasta de resultados, o projeto também armazena:

Uma cópia do código-fonte utilizado na execução
Um gráfico de comparação geral
Um log com a data e o horário da execução
⚠️ Observações
O dataset não está incluído no repositório
Recomenda-se o uso de GPU para melhor desempenho
Os resultados são organizados automaticamente para facilitar a análise de cada execução

## 🖥️ Ambiente utilizado

- **Sistema Operacional:** Windows 10 (10.0.19045.5435)
- **Python:** 3.7.9
- **GPU:** NVIDIA GeForce RTX 3050 Mobile
- **Driver NVIDIA:** 576.52
- **CUDA Toolkit:** 11.2
- **nvcc:** V11.2.67
- **cuDNN:** 8.2.1
- **CUDA suportado pelo driver:** 12.9
