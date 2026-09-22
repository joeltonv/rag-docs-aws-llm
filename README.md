# Assistente Virtual RAG para Suporte Técnico

*Desenvolvimento e avaliação de um assistente virtual baseado em RAG para recuperação de conhecimento e suporte técnico.*

Pipeline local para extração, normalização estrutural, ingestão e vetorização de documentação técnica (DraftSight e SOLIDWORKS), integrando recuperação semântica e geração de respostas com modelos fundacionais via AWS Bedrock.

O repositório fornece o ciclo completo de RAG: processamento offline de manuais extraídos de arquivos de ajuda (.chm/HTML), persistência relacional com rastreabilidade, indexação vetorial, API REST em Flask, interface de exploração/benchmark em Streamlit e um pipeline experimental automatizado com avaliação via LLM-as-a-Judge.

---

## Arquitetura da Solução


```

[Docs HTML/XML/TXT]
│
▼ (converter_documentacao.py)
[Markdown Limpo (docs_markdown/)]
│
▼ (pipeline_ingestao_rag.py)
[SQLite (knowledge_base.db)] ─── Metadados, hierarquia e integridade por hash SHA-256
│
▼ (gerar_embeddings.py - BGE-M3 local)
[ChromaDB (chroma_db/)] ─────── Índice vetorial persistente
│
├──────────────────────────────────────────┐
▼                                          ▼
[API Flask (app.py)]                      [Pipeline Experimental]
├── GET  /health                          ├── dataset_builder.py
├── GET  /search (com tradução opcional)  ├── avaliador_experimental.py
└── POST /chat (RAG + AWS Bedrock)        └── metrics_pipeline.py
│                                          │
▼                                          ▼
[Streamlit UI (streamlit_app/)]           [Relatórios & Figuras (results/)]

```

### Componentes Principais

1. **Conversão e Higienização (`converter_documentacao.py`):**
   - Varre recursivamente arquivos `.htm`, `.html`, `.xhtml`, `.xml` e `.txt`.
   - Remove tags de ruído estrutural (`nav`, `header`, `footer`, `aside`, `script`, breadcrumbs e menus).
   - Preserva tabelas (com resolução de `rowspan`/`colspan` ou formato chave-valor) e blocos `<pre>` como fences Markdown atômicas.
   - Suporte a multiprocessamento (`--convert-workers`).

2. **Ingestão Estruturada (`pipeline_ingestao_rag.py`):**
   - Segmentação orientada pela hierarquia de cabeçalhos (Markdown e numéricos).
   - Chunking com limite configurável de caracteres e sobreposição (*overlap*) de sentenças via NLTK Punkt.
   - Preserva blocos de código e tabelas sem quebras arbitrárias.
   - Armazena metadados em SQLite (`knowledge_base.db`) com integridade referencial (`ON DELETE CASCADE`) e hashes SHA-256 para idempotência.

3. **Vetorização e Indexação (`gerar_embeddings.py`):**
   - Utiliza SentenceTransformers local com modelo `BAAI/bge-m3` (1024 dimensões) por padrão.
   - IDs estáveis gerados por hash SHA-256 da rota do documento + índice do chunk.
   - Persistência em lote (`batch_size`) em coleção ChromaDB (`manuais_treinamento`).

4. **Camada de Serviço RAG e LLMs (`rag_service.py` e `bedrock_service.py`):**
   - **Recuperação:** Busca vetorial no ChromaDB com injeção de metadados contextuais (software, hierarquia e título).
   - **Tradução de Consulta:** Tradução opcional e contextual de termos técnicos PT-BR para inglês antes da busca vetorial via Bedrock.
   - **Invocação AWS Bedrock:**
     - Suporte nativo à API `Converse` com retry exponencial e detecção de erros transientes.
     - Suporte específico para a família Meta Llama 4 via payload nativo (`<|begin_of_text|>...`) com fallback cross-region automático (`us.`).
     - Adequação automática de hiperparâmetros (ex.: omissão de `temperature` para modelos Claude Sonnet 5).
     - Extração de tokens e métricas de latência.

5. **Interface e Serviços HTTP:**
   - **Flask (`app.py`):** Endpoints REST JSON `/health`, `/search` e `/chat`.
   - **Streamlit (`streamlit_app/`):** Interface visual desacoplada para consultas ad-hoc e benchmark comparativo lado a lado de modelos.

---

## Estrutura de Diretórios

```text
TCC_Pos_LLMs/
├── app.py                           # API REST Flask
├── bedrock_service.py               # Integração e clientes de inferência AWS Bedrock
├── rag_service.py                   # Orquestrador de busca vetorial, contexto e chat
├── converter_documentacao.py        # Conversão de HTML/documentos para Markdown
├── pipeline_ingestao_rag.py         # Chunking estruturado e persistência SQLite
├── gerar_embeddings.py              # Vetorização e carga no ChromaDB
├── executar_pipeline_documentacao.py# CLI para execução ponta a ponta do pipeline
├── dataset_builder.py               # Geração do dataset padrão-ouro (heurística/LLM)
├── avaliador_experimental.py        # Execução de experimentos em lote (variantes x LLMs)
├── metrics_pipeline.py              # Consolidação, LLM-as-a-Judge e relatórios
├── run_once_download.py             # Download inicial dos modelos de embeddings
├── testar_leitura.py                # Diagnóstico e verificação da base SQLite
├── experiments/
│   ├── config.py                    # Configuração central dos experimentos
│   ├── dataset_generation.py        # Algoritmos de extração e balanceamento do dataset
│   ├── evaluation.py                # Loop de execução de testes de inferência
│   └── metrics_reporting.py         # Métricas de retrieval, julgamento e plots
├── docs/                            # Documentação bruta de origem (não versionada)
├── docs_markdown/                   # Documentos intermediários em Markdown
├── modelos/                         # Cache local dos modelos SentenceTransformers
│   └── bge-m3/
├── datasets/                        # Datasets padrão-ouro gerados
├── results/                         # Resultados brutos e relatórios comparativos
├── requirements.txt                 # Dependências do projeto
└── README.md

```

---

## Instalação e Requisitos

### Pré-requisitos

* Python 3.11 ou superior
* Credenciais AWS configuradas no ambiente (`~/.aws/credentials` ou variáveis de ambiente) com permissão para `bedrock:InvokeModel` nas regiões utilizadas.

### 1. Configurar Ambiente Virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Baixar Modelos Locais de Embeddings

O pipeline depende da presença do modelo BGE-M3 local em `modelos/bge-m3`. Execute o utilitário de download:

```powershell
python run_once_download.py

```

---

## Execução do Pipeline de Dados

O pipeline completo pode ser executado de forma unificada ou em etapas individuais.

### Execução Unificada

```powershell
python executar_pipeline_documentacao.py docs `
  --markdown-output docs_markdown `
  --db knowledge_base.db `
  --chroma-dir chroma_db `
  --convert-workers 4 `
  --max-chunk-chars 1200 `
  --chunk-overlap-sentences 1 `
  --embedding-device cpu `
  --embedding-batch-size 500

```

### Execução Modular

1. **Conversão HTML para Markdown:**
```powershell
python -c "from pathlib import Path; from converter_documentacao import convert_documentation_tree; convert_documentation_tree(Path('docs'), Path('docs_markdown'), max_workers=4)"

```


2. **Ingestão e Chunking no SQLite:**
```powershell
python pipeline_ingestao_rag.py docs_markdown --db knowledge_base.db --max-chunk-chars 1200 --chunk-overlap-sentences 1

```


3. **Geração de Embeddings e Carga no ChromaDB:**
```powershell
python gerar_embeddings.py --sqlite-path knowledge_base.db --chroma-dir chroma_db --device cpu --batch-size 500

```


4. **Auditoria da Base Indexada:**
```powershell
python testar_leitura.py --db knowledge_base.db --search "extrusão"

```



---

## Execução das Aplicações

### API REST Flask

Inicia o serviço HTTP na porta 5000:

```powershell
python app.py

```

#### Endpoints Principais:

* `GET /health`: Status da coleção vetorial, total de chunks indexados e modelo ativo.
* `GET /search?q=<consulta>&top_k=5&translate_query=true`: Recuperação vetorial pura e montagem de contexto.
* `POST /chat`: Execução completa do fluxo RAG.
```json
{
  "question": "Como criar uma restrição de paralelismo no esboço?",
  "top_k": 5,
  "model_id": "deepseek.v3.2",
  "translate_query": true
}

```



### Interface Streamlit

Com a API Flask em execução em outro terminal:

```powershell
streamlit run streamlit_app/app.py

```

A interface permite testar perguntas isoladas, alterar hiperparâmetros em tempo de execução e submeter a mesma consulta para múltiplos LLMs em paralelo via endpoint `/chat`.

---

## Avaliação Experimental e LLM-as-a-Judge

O projeto inclui um framework experimental automatizado para medir ganhos de recuperação e geração com rigor metodológico.

### 1. Gerar Dataset Padrão-Ouro

Extrai seções balanceadas por categoria (`procedimento`, `diagnóstico`, `configuração`, etc.) e gera pares pergunta/resposta ideal:

```powershell
python dataset_builder.py `
  --docs-root docs_markdown `
  --output datasets/dataset_padrao_ouro.json `
  --target-size 50 `
  --seed 42 `
  --use-llm-questions `
  --question-model-id us.anthropic.claude-sonnet-5

```

### 2. Executar Inferência em Lote

Avalia as 4 variantes experimentais de recuperação em relação aos modelos de chat configurados:

```powershell
python avaliador_experimental.py `
  --dataset datasets/dataset_padrao_ouro.json `
  --results results/resultados_experimento.csv `
  --top-k 10

```

**Variantes avaliadas por padrão:**

* `baseline`: Top-K = 4, sem tradução de consulta.
* `baseline_traduzido`: Top-K = 4, com tradução PT-BR -> EN.
* `topk_aumentado`: Top-K = 10, sem tradução de consulta.
* `topk_aumentado_traduzido`: Top-K = 10, com tradução PT-BR -> EN.

### 3. Consolidar Métricas e Relatórios

Calcula métricas de ranking, executa o juiz automatizado via Tool Use com esquema estrito e gera visualizações:

```powershell
python metrics_pipeline.py `
  --dataset datasets/dataset_padrao_ouro.json `
  --results results/resultados_experimento.csv

```

**Artefatos gerados em `results/analise_experimento/`:**

* `relatorio_retrieval.md`: Análise de Hit Rate@K, Recall@K, Precision@K, MRR e similaridade média.
* `relatorio_generation.md`: Pontuações do LLM-juiz (escala 1 a 5) em Groundedness, Corretude, Completude, Clareza, Precisão Técnica e Ausência de Alucinação.
* `relatorio_comparativo.md`: Ganho percentual e delta absoluto em relação à baseline.
* `figures/`: Gráficos de barras, boxplots de latência/tokens, curvas de precisão/recall e matrizes de correlação em `.png` e `.svg`.

---

## Configuração por Variáveis de Ambiente

As principais configurações do sistema são parametrizadas via variáveis de ambiente:

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `AWS_REGION` | `us-east-1` | Região AWS utilizada para chamadas ao Amazon Bedrock |
| `RAG_CHROMA_DIR` | `chroma_db` | Diretório de persistência do banco vetorial ChromaDB |
| `RAG_COLLECTION_NAME` | `manuais_treinamento` | Nome da coleção no ChromaDB |
| `RAG_TOP_K` | `10` | Quantidade padrão de chunks recuperados |
| `RAG_MAX_CONTEXT_CHARS` | `6000` | Limite de caracteres para injeção de contexto no prompt |
| `RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Provedor de embeddings (`sentence-transformers` ou `bedrock`) |
| `RAG_LOCAL_EMBEDDING_MODEL_PATH` | `modelos/bge-m3` | Caminho do modelo SentenceTransformers local |
| `RAG_LOCAL_EMBEDDING_DEVICE` | `cpu` | Dispositivo de computação dos embeddings (`cpu` ou `cuda`) |
| `RAG_BEDROCK_CHAT_MODEL_ID` | `deepseek.v3.2` | Modelo de chat padrão do Bedrock |
| `RAG_QUERY_TRANSLATION_ENABLED` | `true` | Ativa a tradução contextual da consulta para inglês |
| `RAG_QUERY_TRANSLATION_MODEL_ID` | `deepseek.v3.2` | Modelo utilizado para a tradução da consulta |
| `EXPERIMENT_BASELINE_TOP_K` | `4` | Top-K de referência nos testes experimentais |
| `EXPERIMENT_TOP_K` | `10` | Top-K expandido nos testes experimentais |
| `EXPERIMENT_JUDGE_MODEL_ID` | `us.anthropic.claude-sonnet-5` | Modelo utilizado como LLM-as-a-Judge |

---

## Esquema Relacional (SQLite)

O banco relacional `knowledge_base.db` mantém a rastreabilidade entre o documento fonte e seus chunks:

```sql
CREATE TABLE softwares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome_software TEXT,
    nome_arquivo TEXT,
    source_path TEXT UNIQUE,     -- Chave lógica para idempotência
    arquivo_hash TEXT,           -- Hash SHA-256 do arquivo original
    ingestao_em TEXT             -- Timestamp UTC ISO-8601
);

CREATE TABLE topicos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    software_id INTEGER,
    source_path TEXT,
    chunk_ordem INTEGER,
    subchunk_ordem INTEGER,
    subchunk_total INTEGER,
    hierarquia TEXT,             -- Trilha de tópicos (ex.: Seção > Subseção)
    titulo TEXT,
    conteudo_texto TEXT,
    conteudo_hash TEXT,          -- Hash SHA-256 do texto do chunk
    ingestao_em TEXT,
    FOREIGN KEY (software_id) REFERENCES softwares (id) ON DELETE CASCADE
);

```

---

## Testes Automatizados

Para rodar a suíte completa de testes unitários e de integração:

```powershell
pytest -v

```

Os testes validam:

* Integridade na conversão HTML para Markdown (estruturas complexas de tabela e listas).
* Algoritmo de quebra e fallback de chunks textuais.
* Idempotência e integridade referencial nas inserções do SQLite.
* Geração determinística de IDs vetoriais.
* Contrato da API Flask (`/health`, `/search`, `/chat`).
