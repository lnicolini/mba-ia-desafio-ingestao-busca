# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de **RAG (Retrieval-Augmented Generation)** que permite fazer perguntas sobre um PDF usando LangChain, Google Gemini e PostgreSQL com pgVector.

## 🎯 O que esta aplicação faz?

Esta aplicação implementa um sistema de chat inteligente baseado em RAG que permite fazer perguntas sobre documentos PDF:

1. **📄 Ingestão de PDF**: Carrega e processa documentos PDF, dividindo-os em chunks menores
2. **🧠 Vetorização**: Gera embeddings (representações vetoriais) do conteúdo usando Google Gemini
3. **💾 Armazenamento**: Salva os vetores no PostgreSQL com pgVector para busca eficiente
4. **🔍 Busca Semântica**: Encontra os trechos mais relevantes do documento baseado no significado da pergunta
5. **🤖 Geração de Resposta**: Usa o contexto encontrado para gerar respostas precisas via LLM (Gemini)
6. **✅ Validação de Contexto**: Responde apenas com base no conteúdo do PDF, evitando alucinações

## 📋 Requisitos

- Python 3.10+
- Docker & Docker Compose
- Conta Google Cloud com API Key do Gemini

## 🚀 Como Executar

### 1. Clone o repositório e navegue até a pasta do projeto

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure sua **API Key do Google Gemini**:

```env
# Google Gemini API Configuration
GOOGLE_API_KEY=sua_chave_api_aqui

# Demais configurações já estão prontas para uso
```

**Como obter a API Key do Google Gemini:**

1. Acesse: <https://aistudio.google.com/app/apikey>
2. Clique em "Create API Key"
3. Copie a chave gerada e cole no arquivo `.env`

### 3. Crie e ative um ambiente virtual Python

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate

```

### 4. Instale as dependências Python

```bash
pip install -r requirements.txt
```

### 5. Suba o banco de dados PostgreSQL com pgVector

```bash
docker-compose up -d
```

Aguarde o banco inicializar completamente (~10 segundos).

### 4. Ingerir o PDF no banco de dados

Coloque seu arquivo PDF na raiz do projeto (ou ajuste a variável `PDF_PATH` no `.env`).

Execute o script de ingestão:

```bash
python src/ingest.py
```

**Saída esperada:**

```bash
============================================================
INICIANDO PROCESSO DE INGESTÃO
============================================================
PDF: /caminho/para/document.pdf
Chunk Size: 1000
Chunk Overlap: 150
Embedding Model: models/embedding-001
============================================================

[1/4] Carregando PDF...
✓ PDF carregado com sucesso: 5 página(s)

[2/4] Dividindo documento em chunks...
✓ Documento dividido em 42 chunk(s)

[3/4] Preparando documentos para ingestão...
✓ 42 documento(s) preparado(s)

[4/4] Gerando embeddings e salvando no PostgreSQL...
✓ 42 chunk(s) armazenado(s) com sucesso no banco de dados!

============================================================
INGESTÃO CONCLUÍDA COM SUCESSO!
============================================================

Agora você pode executar o chat.py para fazer perguntas sobre o PDF.
```

### 5. Executar o chat interativo

```bash
python src/chat.py
```

## 💬 Exemplos de Uso

**Pergunta dentro do contexto do PDF:**

```bash
Faça sua pergunta:

PERGUNTA: Qual o faturamento da empresa Alfa IA Indústria ?

Processando sua pergunta... ✓

RESPOSTA: R$ 548.789.613,65

------------------------------------------------------------
```

**Pergunta fora do contexto:**

```bash
Faça sua pergunta:

PERGUNTA: Qual a capital do Brasil?

Processando sua pergunta... ✓

RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

------------------------------------------------------------
```

**Para sair do chat:**

```bash
Faça sua pergunta:

PERGUNTA: sair

Encerrando chat. Até logo! 👋
```

## 🏗️ Arquitetura

### Separação de Responsabilidades

O projeto está organizado com clara separação de responsabilidades:

- **`search.py`**: Contém toda a lógica de busca vetorial e RAG com LLM
  - Conexão com pgVector
  - Busca de documentos similares
  - Configuração da chain RAG (prompt + LLM)
  - Função `ask_question()` que executa o fluxo RAG completo
  - Pode ser executado standalone para testes

- **`chat.py`**: Interface CLI para interação com o usuário
  - Loop interativo de perguntas/respostas
  - Validação de entrada
  - Chamada para `ask_question()` de `search.py`
  - Formatação da saída

- **`ingest.py`**: Script de ingestão do PDF
  - Carregamento do PDF
  - Divisão em chunks
  - Geração de embeddings
  - Armazenamento no pgVector

### Fluxo de Ingestão (`ingest.py`)

1. **Carregamento**: Lê o PDF usando `PyPDFLoader`
2. **Split**: Divide em chunks de 1000 caracteres com overlap de 150
3. **Embedding**: Gera vetores usando `models/embedding-001` do Gemini
4. **Storage**: Armazena no PostgreSQL com pgVector

### Fluxo de Consulta (`chat.py` → `search.py`)

1. **Input**: `chat.py` recebe pergunta do usuário via CLI
2. **Delegação**: Chama `ask_question()` de `search.py`
3. **Vetorização**: `search.py` converte pergunta em embedding
4. **Search**: Busca top-10 chunks mais similares (k=10)
5. **Context**: Concatena os chunks encontrados
6. **Prompt**: Monta prompt com contexto + regras + pergunta
7. **LLM**: Chama `gemini-2.5-flash-lite` para gerar resposta
8. **Output**: `chat.py` exibe resposta formatada ao usuário

## 📦 Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain** - Framework para aplicações com LLMs
- **Google Gemini** - Modelo de embeddings e LLM
- **PostgreSQL 17** - Banco de dados relacional
- **pgVector** - Extensão para busca vetorial
- **Docker** - Containerização do banco de dados

## 📁 Estrutura do Projeto

```
├── document.pdf             # PDF para análise
├── docker-compose.yml       # Configuração do PostgreSQL + pgVector
├── .env                     # Variáveis de ambiente (não commitado)
├── .env.example             # Template de configuração
├── requirements.txt         # Dependências Python
├── README.md                # Documentação do projeto
└── src/ 
    ├── ingest.py            # Script de ingestão do PDF
    ├── search.py            # Lógica de busca vetorial e RAG
    └── chat.py              # Interface CLI interativa
```

## 📝 Notas

- O modelo `gemini-2.5-flash-lite` é usado com `temperature=0.0` para respostas mais determinísticas
- O sistema busca os 10 chunks mais relevantes (k=10) para cada pergunta
- O prompt força a LLM a responder apenas com base no contexto fornecido
- Perguntas fora do escopo retornam: "Não tenho informações necessárias para responder sua pergunta."

Este projeto foi desenvolvido para o MBA de Engenharia de Software com IA - Full Cycle.
