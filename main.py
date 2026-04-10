from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional
import asyncio
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = api_key
)

class Estado_Pokemon(TypedDict):
    pokemons_adversarios: str
    preferencia_usuario: str
    analise_adversario: str
    time_sugerido: str
    explicacao: str

prompt_analise = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um analista especialista em batalhas Pokémon no Pixelmon."
     "Sabendo o time adversário, identifique: tipos predominantes, ameaças principais, fraquezas exploráveis e estratégia geral do oponente."
     "Seja técnico e direto."
     "Responda em Português."),

     ("human", "Time adversário: {pokemons_adversarios}")
])

prompt_montar_time = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um especialista em montagem de times de Pokémon NO PIXELMON."
     "Com base da análise do adversário, monte o melhor time de 6 Pokémon para enfrentá-lo."
     "Se o usuário pediu um Pokémon específico, inclua-o obrigatoriamente, sem questionamentos."
     "Liste apenas os 6 nomes do time, um por linha, sem explicações de nada por enquanto."
     "Responda em português."),

     ("human", 
      "Análise do adversário: {analise_adversario}\n"
      "Pokémon obrigatório no time (se houver): {preferencia_usuario}\n"
      "Monte o time ideal: ")
])

prompt_explicacao = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um comentarista de batalhas Pokémon no pixelmon."
     "Explique de forma clara por que cada Pokémon foi escolhido, quais as sinergias entre eles e como contra-atacar a equipe adversária."
     "Se um Pokémon foi escolha do treinador, destaque como ele foi encaixado estratégicamente."
     "Responda em português."),
     ("human",
      "Time adversário: {pokemons_adversarios}\n"
      "Análise do time adversário: {analise_adversario}\n"
      "Time montado: {time_sugerido}\n"
      "Explique as sinergias: ")
])


cadeia_analise = prompt_analise | modelo | StrOutputParser()
cadeia_montar_time = prompt_montar_time | modelo | StrOutputParser()
cadeia_sinergias = prompt_explicacao | modelo | StrOutputParser()


async def no_analisar_adversario (estado: Estado_Pokemon, config=RunnableConfig):
    """Nó 1 - Analisa tipos, ameaças e fraquezas do time adversario. """
    print("\nAnalisando o time adversário...")
    analise = await cadeia_analise.ainvoke({"pokemons_adversarios": estado["pokemons_adversarios"]}, config)
    
    return {"analise_adversario": analise}

async def no_montar_time(estado: Estado_Pokemon, config = RunnableConfig):
    """Nó 2 - Monta o time de 6 Pokémon, respeitando a preferência do treinador."""
    print("\nMontando o time perfeito...")
    time = await cadeia_montar_time.ainvoke(
        {
            "analise_adversario": estado["analise_adversario"],
            "preferencia_usuario": estado.get("preferencia_usuario") or "Nenhum",
        }, config,)
    
    return {"time_sugerido": time}

async def no_explicar(estado: Estado_Pokemon, config = RunnableConfig):
    """Nó 3 - Explica as sinergias e a estratégia utilizada para montar o time"""
    print("Explicnado as Sinergias do time montado: ")
    explicacao = await cadeia_sinergias.ainvoke(
        {
            "pokemons_adversarios": estado["pokemons_adversarios"],
            "analise_adversario": estado ["analise_adversario"],
            "time_sugerido": estado ["time_sugerido"],
        }, config,)
    
    return {"explicacao": explicacao}


grafo = StateGraph(Estado_Pokemon)

grafo.add_node("analisar_adversario", no_analisar_adversario)
grafo.add_node("montar_time", no_montar_time)
grafo.add_node("explicar", no_explicar)

grafo.add_edge(START, "analisar_adversario")
grafo.add_edge("analisar_adversario", "montar_time")
grafo.add_edge("montar_time", "explicar")
grafo.add_edge("explicar", END)

app = grafo.compile()

async def main():
    print("-" * 55)
    print("     ASSISTENTE DE TREINADOR")
    print("-" * 55)

    print("Digite os Pokémon do adversário(1 a 6): ")
    pokemons = []
    for i in range (1,7):
        pokemon = input (f"  ->{i}. ").strip()
        if not pokemon:
            break
        pokemons.append(pokemon)

    adversario =", ".join(pokemons)

    print("Você quer algum Pokémon específico no seu time? (deixe em branco para pular)")
    preferencia = input("-> ").strip()

    print()

    resultado = await app.ainvoke({
        "pokemons_adversarios": adversario,
        "preferencia_usuario": preferencia,
        "analise_adversario": "",
        "time_sugerido": "",
        "explicacao": "",
    })

    print("\n" + "-" * 55)
    print("     ⚔️ ANÁLISE DO ADVERSÁRIO")
    print("-" * 55)
    print(resultado["analise_adversario"])

    print("\n" + "-" * 55)
    print("     🫱🏼 SEU TIME IDEAL")
    print("-" * 55 )
    print(resultado["time_sugerido"])

    print("\n" + "-" * 55)
    print("     🎯 EXPLICAÇÃO E ESTRATÉGIAS")
    print("-" * 55 )
    print(resultado["explicacao"])
    print("-" * 55 )

if __name__ == "__main__":
    asyncio.run(main())