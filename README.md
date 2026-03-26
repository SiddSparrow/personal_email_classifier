# Email Classifier

Pipeline automatizado que monitora emails no Gmail, classifica devolutivas positivas de processos seletivos usando NLP e notifica via Telegram.

## Como funciona

1. **Leitura** - Busca emails nao lidos no Gmail via API
2. **Deduplicacao** - Ignora emails ja processados (state.json)
3. **Preprocessamento** - Limpa HTML e remove URLs
4. **Classificacao** - Sentence-Transformer Embeddings + SMOTE + Logistic Regression identifica devolutivas positivas
5. **Notificacao** - Envia alerta no Telegram com preview e link direto pro Gmail

## Estrutura

```
email_classifier/
├── main.py                          # Orquestrador do pipeline
├── core/
│   ├── interfaces.py                # Contratos (ABC)
│   └── config.py                    # Configuracao e logging
├── reader/
│   └── gmail_reader.py              # Leitura de emails via Gmail API
├── preprocessing/
│   └── preprocessor.py              # Limpeza de texto (HTML strip, URLs)
├── classification/
│   ├── classifier.py                # Classificador (Embeddings + LogisticRegression)
│   └── trainer.py                   # Treinamento do modelo (SMOTE + Embeddings)
├── notification/
│   └── notifier.py                  # Notificacoes via Telegram
├── state/
│   └── state_manager.py             # Persistencia de emails processados
├── data/
│   ├── train_data/
│   │   ├── job_offers.txt           # Exemplos de devolutivas positivas
│   │   └── other.txt                # Exemplos de emails comuns
│   └── state.json                   # IDs de emails ja processados
├── models/
│   └── model.pkl                    # Modelo treinado (LogisticRegression)
├── credentials/
│   └── token.json                   # Token OAuth Gmail (gerado automaticamente)
└── logs/
    └── classifier.log
```

## Configuracao

### 1. Dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Gmail API

- Crie um projeto no [Google Cloud Console](https://console.cloud.google.com/)
- Ative a **Gmail API**
- Crie credenciais OAuth 2.0 (tipo Desktop)
- Salve o arquivo como `credentials.json` na raiz do projeto

### 3. Telegram Bot

- Crie um bot via [@BotFather](https://t.me/BotFather) e copie o token
- Envie `/start` para o bot
- Descubra seu `chat_id` acessando `https://api.telegram.org/bot<TOKEN>/getUpdates`

### 4. Variaveis de ambiente

Copie o exemplo e preencha:

```bash
cp .env.example .env
```

Edite o `.env` com seu `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`.

## Uso

### Executar o pipeline

```bash
python main.py
```

### Retreinar o modelo

Edite os arquivos em `data/train_data/` e execute:

```bash
python -m classification.trainer
```

Na primeira execucao, o modelo `paraphrase-multilingual-MiniLM-L12-v2` (~100MB) sera baixado do HuggingFace Hub e cacheado localmente.

### Automacao com crontab (VPS)

```cron
0 * * * * cd /var/www/email_classifier && /var/www/email_classifier/venv/bin/python main.py >> /var/www/email_classifier/logs/cron.log 2>&1
```

## Modelo

- **Embeddings**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) - vetores semanticos de 384 dimensoes, multilingue (PT + EN)
- **Balanceamento**: SMOTE gera amostras sinteticas da classe minoritaria no espaco vetorial
- **Classificador**: Logistic Regression com `class_weight='balanced'`
- **Classes**: `job_offer` (devolutivas positivas) e `other`
- **Threshold**: 75% de confianca para disparar notificacao (configuravel via `CONFIDENCE_THRESHOLD`)

---

# Email Classifier (EN)

Automated pipeline that monitors Gmail emails, classifies positive recruitment responses using NLP, and sends notifications via Telegram.

## How it works

1. **Reading** - Fetches unread emails from Gmail via API
2. **Deduplication** - Skips already processed emails (state.json)
3. **Preprocessing** - Strips HTML and removes URLs
4. **Classification** - Sentence-Transformer Embeddings + SMOTE + Logistic Regression identifies positive recruitment responses
5. **Notification** - Sends Telegram alert with preview and direct Gmail link

## Structure

```
email_classifier/
├── main.py                          # Pipeline orchestrator
├── core/
│   ├── interfaces.py                # Contracts (ABC)
│   └── config.py                    # Configuration and logging
├── reader/
│   └── gmail_reader.py              # Email reading via Gmail API
├── preprocessing/
│   └── preprocessor.py              # Text cleanup (HTML strip, URLs)
├── classification/
│   ├── classifier.py                # Classifier (Embeddings + LogisticRegression)
│   └── trainer.py                   # Model training (SMOTE + Embeddings)
├── notification/
│   └── notifier.py                  # Telegram notifications
├── state/
│   └── state_manager.py             # Processed emails persistence
├── data/
│   ├── train_data/
│   │   ├── job_offers.txt           # Positive recruitment response examples
│   │   └── other.txt                # Common email examples
│   └── state.json                   # Already processed email IDs
├── models/
│   └── model.pkl                    # Trained model (LogisticRegression)
├── credentials/
│   └── token.json                   # Gmail OAuth token (auto-generated)
└── logs/
    └── classifier.log
```

## Setup

### 1. Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Gmail API

- Create a project on [Google Cloud Console](https://console.cloud.google.com/)
- Enable the **Gmail API**
- Create OAuth 2.0 credentials (Desktop type)
- Save the file as `credentials.json` in the project root

### 3. Telegram Bot

- Create a bot via [@BotFather](https://t.me/BotFather) and copy the token
- Send `/start` to the bot
- Find your `chat_id` at `https://api.telegram.org/bot<TOKEN>/getUpdates`

### 4. Environment variables

Copy the example and fill in:

```bash
cp .env.example .env
```

Edit `.env` with your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Usage

### Run the pipeline

```bash
python main.py
```

### Retrain the model

Edit files in `data/train_data/` and run:

```bash
python -m classification.trainer
```

On first run, the `paraphrase-multilingual-MiniLM-L12-v2` model (~100MB) will be downloaded from HuggingFace Hub and cached locally.

### Automation with crontab (VPS)

```cron
0 * * * * cd /var/www/email_classifier && /var/www/email_classifier/venv/bin/python main.py >> /var/www/email_classifier/logs/cron.log 2>&1
```

## Model

- **Embeddings**: Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) - 384-dimensional semantic vectors, multilingual (PT + EN)
- **Balancing**: SMOTE generates synthetic samples of the minority class in the vector space
- **Classifier**: Logistic Regression with `class_weight='balanced'`
- **Classes**: `job_offer` (positive recruitment responses) and `other`
- **Threshold**: 75% confidence to trigger notification (configurable via `CONFIDENCE_THRESHOLD`)
