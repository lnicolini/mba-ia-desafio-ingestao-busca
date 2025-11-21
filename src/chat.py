"""
Interface CLI para Chat com RAG (Retrieval-Augmented Generation)

Este script fornece uma interface interativa de linha de comando
para fazer perguntas sobre o conteúdo do PDF ingerido.

A lógica de busca e resposta está implementada em search.py
"""

import os
from dotenv import load_dotenv
from search import ask_question

# Carregar variáveis de ambiente
load_dotenv()

# Validar variáveis obrigatórias
required_vars = ["GOOGLE_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente {var} não está configurada")


def main():
    """Função principal do chat CLI"""
    
    print("=" * 60)
    print("CHAT RAG - Sistema de Perguntas e Respostas")
    print("=" * 60)
    print("\nVocê pode fazer perguntas sobre o PDF ingerido.")
    print("Digite 'sair' para encerrar o chat.\n")
    print("=" * 60)
    
    # Loop principal do chat
    while True:
        try:
            # Receber pergunta do usuário
            pergunta = input("\nFaça sua pergunta:\n\nPERGUNTA: ").strip()
            
            # Verificar se quer sair
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\nEncerrando chat. Até logo! 👋")
                break
            
            # Verificar se a pergunta não está vazia
            if not pergunta:
                print("Por favor, digite uma pergunta válida.")
                continue
            
            # Buscar e responder usando o módulo search
            print("\nProcessando sua pergunta...", end=" ", flush=True)
            resposta = ask_question(pergunta)
            print("✓\n")
            
            # Exibir resposta
            print(f"RESPOSTA: {resposta}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nEncerrando chat. Até logo! 👋")
            break
        except Exception as e:
            print(f"\n✗ Erro ao processar pergunta: {e}\n")
            continue


if __name__ == "__main__":
    main()
