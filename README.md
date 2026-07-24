# BERT-CPU

## Requisitos

- Python >= 3.8
- NumPy

## Setup

Crie um ambiente virtual e instale as dependências

```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)

# Instalar as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

## Rodar Experimentos

Baseline (ReLU, 100 épocas, LR=1x10^-2)
```bash
python -m exercises.task_binary_classification
```

Selecione a função de ativação desejada

```bash
# Rodar com Learnable Activation sem restrição (Experimento 2)
python -m exercises.task_binary_classification --activation learnable
```

```bash
# Rodar com Learnable Activation normalizada (Experimento 3)
python -m exercises.task_binary_classification --activation normalized
```
---

Selecione a quantidade de épocas

```bash
# Rodar com 25 épocas (Experimento 4)
python -m exercises.task_binary_classification --epochs 25
```

```bash
# Rodar com 50 épocas (Experimento 5)
python -m exercises.task_binary_classification --epochs 50
```

```bash
# Rodar com 200 épocas (Experimento 6)
python -m exercises.task_binary_classification --epochs 200
```

```bash
# Rodar com 400 épocas (Experimento 7)
python -m exercises.task_binary_classification --epochs 400
```
---

Selecione a taxa de aprendizado

```bash
# Rodar com lr=1x10^-5 (Experimento 8)
python -m exercises.task_binary_classification --lr 1e-5
```

```bash
# Rodar com lr=1x10^-3 (Experimento 9)
python -m exercises.task_binary_classification --lr 1e-3
```

```bash
# Rodar com lr=5x10^-2 (Experimento 10)
python -m exercises.task_binary_classification --lr 5e-2
```

```bash
# Rodar com lr=1x10^-1 (Experimento 11)
python -m exercises.task_binary_classification --lr 1e-1
```
---

Combinações de Learnable Activation com as outras configurações de nº de épocas e learning rates

```bash
# Rodar com Learnable Activation + 25 épocas (Experimento 12)
python -m exercises.task_binary_classification --activation learnable --epochs 25
```

```bash
# Rodar com Learnable Activation + 50 épocas (Experimento 13)
python -m exercises.task_binary_classification --activation learnable --epochs 50
```

```bash
# Rodar com Learnable Activation + 200 épocas (Experimento 14)
python -m exercises.task_binary_classification --activation learnable --epochs 200
```

```bash
# Rodar com Learnable Activation + 400 épocas (Experimento 15)
python -m exercises.task_binary_classification --activation learnable --epochs 400
```

```bash
# Rodar com Learnable Activation + lr=1x10^-5 (Experimento 16)
python -m exercises.task_binary_classification --activation learnable --lr 1e-5
```

```bash
# Rodar com Learnable Activation + lr=1x10^-3 (Experimento 17)
python -m exercises.task_binary_classification --activation learnable --lr 1e-3
```

```bash
# Rodar com Learnable Activation + lr=5x10^-2 (Experimento 18)
python -m exercises.task_binary_classification --activation learnable --lr 5e-2
```

```bash
# Rodar com Learnable Activation + lr=1x10^-1 (Experimento 19)
python -m exercises.task_binary_classification --activation learnable --lr 1e-1
```
---

Combinações de Normalized Learnable Activation com as outras configurações de nº de épocas e learning rates

```bash
# Rodar com Normalized Learnable Activation + 25 épocas (Experimento 20)
python -m exercises.task_binary_classification --activation normalized --epochs 25
```

```bash
# Rodar com Normalized Learnable Activation + 50 épocas (Experimento 21)
python -m exercises.task_binary_classification --activation normalized --epochs 50
```

```bash
# Rodar com Normalized Learnable Activation + 200 épocas (Experimento 22)
python -m exercises.task_binary_classification --activation normalized --epochs 200
```

```bash
# Rodar com Normalized Learnable Activation + 400 épocas (Experimento 23)
python -m exercises.task_binary_classification --activation normalized --epochs 400
```

```bash
# Rodar com Normalized Learnable Activation + lr=1x10^-5 (Experimento 24)
python -m exercises.task_binary_classification --activation normalized --lr 1e-5
```

```bash
# Rodar com Normalized Learnable Activation + lr=1x10^-3 (Experimento 25)
python -m exercises.task_binary_classification --activation normalized --lr 1e-3
```

```bash
# Rodar com Normalized Learnable Activation + lr=5x10^-2 (Experimento 26)
python -m exercises.task_binary_classification --activation normalized --lr 5e-2
```

```bash
# Rodar com Normalized Learnable Activation + lr=1x10^-1 (Experimento 27)
python -m exercises.task_binary_classification --activation normalized --lr 1e-1
```