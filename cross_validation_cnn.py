import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import locale
from openpyxl import Workbook
# Configurar o locale para português (usar vírgula como decimal)
locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8') 

from datetime import datetime
import os
import pandas as pd
import shutil
import sys
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
import time

# Configurações AGGRESSIVAS para GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 1. Aloque toda a memória disponível
        tf.config.experimental.set_memory_growth(gpus[0], False)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Ajuste para sua VRAM
        )
        
        # 2. Configurações de desempenho
        tf.config.optimizer.set_jit(True)  # Ativa XLA
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        # 3. Mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
    except RuntimeError as e:
        print(e)

# Definições
altura = int(os.getenv("ALTURA", 32))
largura = int(os.getenv("LARGURA", 32))
dataset_dir = r"C:\cross_validation_cnn\datasets\rgb"
dataset_name = os.path.basename(dataset_dir)


batch_size = 16
num_folds = 5
epoch = 30
all_data_gen = ImageDataGenerator(rescale=1./255)

# Criar pasta de experimento


def create_experiment_folder(base_name="Exp"):
    folder_name = f"{base_name}_{altura}_{largura}_{dataset_name}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# Construir modelo
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(altura, largura, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# Inicializar logging
experiment_folder = create_experiment_folder()

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

log_path = os.path.join(experiment_folder, "log_terminal.txt")
log_file = open(log_path, "w", buffering=1)

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

codigo_fonte = os.path.abspath(sys.argv[0])
backup_nome = os.path.join(experiment_folder, "backup_codigo.py")
shutil.copy(codigo_fonte, backup_nome)
print(f"Cópia do código salva em: {backup_nome}")

# Carregar imagens e rótulos
filepaths = []
labels = []

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_path, fname))
                labels.append(class_name)

df = pd.DataFrame({'filename': filepaths, 'class': labels})
num_classes = len(df['class'].unique())
class_labels = sorted(df['class'].unique())

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
histories = []
tempos_por_fold = []
metricas_ultimas_epocas = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['class']), 1):
    print(f"\nTreinando fold {fold}/{num_folds}...")

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    train_generator = all_data_gen.flow_from_dataframe(
        df_train,
        x_col='filename',
        y_col='class',
        target_size=(altura, largura),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        subset=None
    )

    val_generator = all_data_gen.flow_from_dataframe(
        df_val,
        x_col='filename',
        y_col='class',
        target_size=(altura, largura),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        subset=None,
    )
    
    print(f"\nDistribuição no Fold {fold}:")
    print("Treino:", df_train['class'].value_counts().sort_index())
    print("Validação:", df_val['class'].value_counts().sort_index())

    model = build_model(num_classes)
    start_time = time.time()
    history = model.fit(train_generator, epochs=epoch, validation_data=val_generator)
    end_time = time.time()

    duracao = end_time - start_time
    print(f"Tempo de treino do Fold {fold}: {duracao:.2f} segundos")
    tempos_por_fold.append(duracao)
    histories.append(history)

    ult_epoch = {
        "fold": fold,
        "loss": history.history['loss'][-1],
        "accuracy": history.history['accuracy'][-1],
        "precision": history.history['precision'][-1],
        "recall": history.history['recall'][-1],
        "val_loss": history.history['val_loss'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_precision": history.history['val_precision'][-1],
        "val_recall": history.history['val_recall'][-1],
        "tempo_treino": duracao
    }
    metricas_ultimas_epocas.append(ult_epoch)

    fold_folder = os.path.join(experiment_folder, f"fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)
    model.save(os.path.join(fold_folder, "modelo_fold.h5"))

    # Gráficos de métricas
    history_dict = history.history
    for metric in ['accuracy', 'precision', 'recall', 'loss']:
        plt.figure()
        plt.plot(history_dict[metric], label=f'{metric.capitalize()} Treino', linestyle='-')
        plt.plot(history_dict[f'val_{metric}'], label=f'{metric.capitalize()} Validação', linestyle='--')
        plt.title(f"{metric.capitalize()} - Fold {fold}")
        plt.xlabel("Épocas")
        plt.ylabel(metric.capitalize())
        plt.xticks(range(len(history_dict[metric])))
        epocas = len(history_dict[metric])
        plt.gca().xaxis.set_major_locator(MultipleLocator(5 if epocas >= 10 else 1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_folder, f"grafico_{metric}_fold_{fold}.png"))
        plt.close()

    # Matriz de confusão
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Matriz de Confusão - Fold {fold}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_folder, f"matriz_confusao_fold_{fold}.png"))
    plt.close()

    generic_labels = [f"modelo_{i+1:02d}" for i in range(len(class_labels))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=generic_labels, yticklabels=generic_labels)
    plt.title(f"Matriz de Confusão (Nomes Genéricos) - Fold {fold}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_folder, f"matriz_confusao_generica_fold_{fold}.png"))
    plt.close()

# Processamento final e salvamento dos resultados
df_metricas_finais = pd.DataFrame(metricas_ultimas_epocas)
colunas = ['fold', 'loss', 'accuracy', 'precision', 'recall', 
           'val_loss', 'val_accuracy', 'val_precision', 'val_recall',
           'tempo_treino']
df_metricas_finais = df_metricas_finais[colunas]

# Nome do arquivo Excel
nome_base = f"Exp_{altura}_{largura}_{dataset_dir}"
nome_excel = os.path.join(experiment_folder, f"{nome_base}.xlsx")

with pd.ExcelWriter(nome_excel) as writer:
    df_metricas_finais.to_excel(writer, sheet_name='Resultados', index=False, float_format="%.6f")
    resumo = df_metricas_finais.describe().loc[['mean', 'std', 'min', 'max']]
    resumo.to_excel(writer, sheet_name='Resumo')
    tempos = pd.DataFrame({
        'Fold': df_metricas_finais['fold'],
        'Tempo (segundos)': df_metricas_finais['tempo_treino'],
        'Tempo (minutos)': df_metricas_finais['tempo_treino']/60
    })
    tempos.to_excel(writer, sheet_name='Tempos', index=False)

print(f"\nResultados salvos em Excel: {nome_excel}")

# Gráfico final comparativo
val_accuracies = df_metricas_finais['val_accuracy']
val_precisions = df_metricas_finais['val_precision']
val_recalls = df_metricas_finais['val_recall']
val_losses = df_metricas_finais['val_loss']
folds = [f"Fold {i}" for i in df_metricas_finais['fold']]

x = np.arange(len(folds))
width = 0.25

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - width, val_accuracies, width=width, label='Validação - Acurácia', color='tab:blue')
ax1.bar(x, val_precisions, width=width, label='Validação - Precisão', color='tab:orange')
ax1.bar(x + width, val_recalls, width=width, label='Validação - Recall', color='tab:green')

ax1.set_xlabel("Fold")
ax1.set_ylabel("Métricas (Acurácia, Precisão, Recall)")
ax1.set_ylim(0, 1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(folds)
ax1.yaxis.set_major_locator(MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.grid(True, axis='y')

ax2 = ax1.twinx()
ax2.plot(x, val_losses, marker='d', color='red', label='Validação - Loss', linewidth=2)
ax2.set_ylabel("Loss")
ax2.set_ylim(0.0, max(val_losses) + 0.1)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))

plt.title("Desempenho por Fold - Acurácia, Precisão, Recall e Loss (Validação)", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(experiment_folder, "comparacao_metricas_por_fold_barras_com_eixo_duplo.png"), 
            bbox_inches='tight', dpi=300)
plt.close()

print("\nCross-validation finalizada com sucesso!")
print(f"Todos os resultados foram salvos em: {experiment_folder}")