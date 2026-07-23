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

Selecione a função de ativação desejada

```bash
# Rodar baseline (Linear -> ReLU -> Linear)
python -m exercises.task_binary_classification

# Rodar com Learnable Activation sem restrição
python -m exercises.task_binary_classification --activation learnable

# Rodar com Learnable Activation normalizada
python -m exercises.task_binary_classification --activation normalized
```
