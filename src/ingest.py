"""
Script de Ingestão de PDF para pgVector

Este script realiza:
1. Carregamento do arquivo PDF
2. Divisão em chunks de texto (1000 caracteres, overlap 150)
3. Geração de embeddings usando Google Gemini
4. Armazenamento dos vetores no PostgreSQL com pgVector
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

# Carregar variáveis de ambiente
load_dotenv()

# Validar variáveis obrigatórias
required_vars = ["GOOGLE_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente {var} não está configurada")

# Configurações
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

def ingest_pdf():
    """Função principal de ingestão do PDF"""
    
    # Resolver caminho do PDF
    if not os.path.isabs(PDF_PATH):
        current_dir = Path(__file__).parent.parent
        pdf_path = current_dir / PDF_PATH
    else:
        pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    print("=" * 60)
    print("INICIANDO PROCESSO DE INGESTÃO")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)

    # 1. Carregar PDF
    print("\n[1/4] Carregando PDF...")
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"✓ PDF carregado com sucesso: {len(docs)} página(s)")

    # 2. Dividir em chunks
    print("\n[2/4] Dividindo documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False
    )
    splits = text_splitter.split_documents(docs)

    if not splits:
        print("✗ Nenhum chunk foi gerado. Verifique o PDF.")
        raise SystemExit(1)

    print(f"✓ Documento dividido em {len(splits)} chunk(s)")

    # 3. Preparar documentos para ingestão
    print("\n[3/4] Preparando documentos para ingestão...")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    # Gerar IDs únicos para cada chunk
    ids = [f"doc-{i:04d}" for i in range(len(enriched))]
    print(f"✓ {len(enriched)} documento(s) preparado(s)")
  #  print(f"  IDs gerados: {', '.join(ids)}")

    # 4. Criar embeddings e armazenar no banco
    print("\n[4/4] Gerando embeddings e salvando no PostgreSQL...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    # Adicionar documentos ao banco
    store.add_documents(documents=enriched, ids=ids)

    print(f"✓ {len(enriched)} chunk(s) armazenado(s) com sucesso no banco de dados!")
    print("\n" + "=" * 60)
    print("INGESTÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print("\nAgora você pode executar o chat.py para fazer perguntas sobre o PDF.")


if __name__ == "__main__":
    ingest_pdf()
