"""
Módulo de busca e resposta com RAG (Retrieval-Augmented Generation)

Este módulo fornece funções para:
1. Buscar documentos relevantes no banco vetorial
2. Gerar respostas usando LLM com base no contexto recuperado
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configurações
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Template do prompt
PROMPT_TEMPLATE = """Você é um assistente que responde perguntas baseado EXCLUSIVAMENTE nas informações fornecidas no CONTEXTO abaixo.
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_vector_store():
    """
    Retorna uma instância configurada do PGVector store
    
    Returns:
        PGVector: Store configurado com embeddings do Gemini
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    
    return store


def search_similar_documents(query: str, k: int = None):
    """
    Busca documentos similares à query fornecida
    
    Args:
        query: String de busca
        k: Número de resultados a retornar (None usa o padrão do env)
    
    Returns:
        Lista de tuplas (documento, score)
    """
    if k is None:
        k = SEARCH_K
    
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return results


def format_search_results(results):
    """
    Formata os resultados da busca para exibição
    
    Args:
        results: Lista de tuplas (documento, score)
    
    Returns:
        String formatada com os resultados
    """
    output = []
    
    for i, (doc, score) in enumerate(results, start=1):
        output.append(f"{'='*60}")
        output.append(f"Resultado {i} (score: {score:.4f})")
        output.append(f"{'='*60}")
        output.append(f"\nTexto:\n{doc.page_content.strip()}")
        
        if doc.metadata:
            output.append(f"\nMetadados:")
            for key, value in doc.metadata.items():
                output.append(f"  {key}: {value}")
        
        output.append("")
    
    return "\n".join(output)


def setup_rag_chain():
    """
    Configura os componentes da chain RAG
    
    Returns:
        tuple: (store, prompt, llm, output_parser)
    """
    # Store vetorial
    store = get_vector_store()
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE
    )
    
    # Prompt template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Parser de output
    output_parser = StrOutputParser()
    
    return store, prompt, llm, output_parser


def ask_question(question: str) -> str:
    """
    Faz uma pergunta usando RAG (busca vetorial + LLM)
    
    Args:
        question: Pergunta a ser respondida
    
    Returns:
        str: Resposta gerada pela LLM
    """
    # Configurar componentes
    store, prompt, llm, output_parser = setup_rag_chain()
    
    # 1. Buscar documentos relevantes
    results = store.similarity_search_with_score(question, k=SEARCH_K)
    
    if not results:
        return "Não tenho informações necessárias para responder sua pergunta."
    
    # 2. Concatenar contexto dos resultados
    context_parts = []
    for i, (doc, score) in enumerate(results, start=1):
        context_parts.append(f"[Trecho {i} - Score: {score:.4f}]:\n{doc.page_content.strip()}\n")
    
    context = "\n".join(context_parts)
    
    # 3. Montar a chain e executar
    chain = prompt | llm | output_parser
    response = chain.invoke({"context": context, "question": question})
    
    return response


if __name__ == "__main__":
    # Exemplo de uso do módulo de busca
    print("=" * 60)
    print("Busca com RAG (Retrieval + LLM)")
    print("=" * 60)
    
    query = input("\nDigite sua pergunta: ")
    
    print("\nProcessando pergunta...", end=" ", flush=True)
    
    try:
        resposta = ask_question(query)
        print("✓\n")
        print(f"RESPOSTA: {resposta}")
    except Exception as e:
        print(f"\n✗ Erro: {e}")
