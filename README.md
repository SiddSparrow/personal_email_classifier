# Email Classifier

Pipeline automatizado que monitora emails no Gmail, classifica devolutivas positivas de processos seletivos usando NLP e notifica via Telegram.

## Como funciona

1. **Leitura** - Busca emails nao lidos no Gmail via API
2. **Deduplicacao** - Ignora emails ja processados (state.json)
3. **Preprocessamento** - Limpa HTML, remove stopwords, aplica stemming (RSLP)
4. **Classificacao** - Modelo Naive Bayes (TF-IDF + MultinomialNB) identifica devolutivas positivas
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
│   └── preprocessor.py              # Pipeline NLP (tokenizacao, stemming)
├── classification/
│   ├── classifier.py                # Classificador Naive Bayes
│   └── trainer.py                   # Treinamento do modelo
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
│   └── model.pkl                    # Modelo treinado
├── credentials/
│   └── token.json                   # Token OAuth Gmail (gerado automaticamente)
└── logs/
    └── classifier.log
```

## Configuracao

### 1. Dependencias

```bash
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

## Modelo

- **Algoritmo**: TF-IDF (bigrams, max 10k features) + Multinomial Naive Bayes
- **Classes**: `job_offer` (devolutivas positivas) e `other`
- **Threshold**: 75% de confianca para disparar notificacao (configuravel via `CONFIDENCE_THRESHOLD`)
