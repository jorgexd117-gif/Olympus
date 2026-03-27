"""Translator example: human reasoning -> structured AI reasoning."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def cargar_modelo_spacy():
    """Load Spanish spaCy model with a clear setup hint if missing."""
    try:
        import spacy
    except Exception as exc:
        raise RuntimeError(
            "spaCy no pudo importarse en este entorno. "
            "Si usas Python 3.14, recrea la venv con Python 3.12 o 3.13."
        ) from exc

    try:
        return spacy.load("es_core_news_sm")
    except OSError as exc:
        raise OSError(
            "No se encontro 'es_core_news_sm'. Ejecuta: "
            "python -m spacy download es_core_news_sm"
        ) from exc


load_dotenv()
load_dotenv(".env.local", override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY no esta definida. Configurala en variables de entorno o .env."
    )

nlp = cargar_modelo_spacy()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def parsear_lenguaje_humano(texto: str) -> dict[str, Any]:
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    verbos = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return {
        "entidades": entidades,
        "acciones": verbos,
        "texto_original": texto,
    }


prompt_template = PromptTemplate.from_template(
    """
Traduce este razonamiento humano a un formato estructurado para IA:
Input humano: {input_humano}
Datos parseados: {parsed_data}

Salida:
- Pasos logicos: [lista de pasos]
- Intencion principal: [resumen]
- Codigo Python sugerido si aplica: [codigo]
"""
)
cadena_traductor = prompt_template | llm | StrOutputParser()

prompt_agente = PromptTemplate.from_template(
    """
Eres un asistente que razona sobre una traduccion estructurada.
Historial de conversacion:
{historial}

Traduccion actual:
{traduccion}

Genera una respuesta razonada y accionable.
"""
)
cadena_agente = prompt_agente | llm | StrOutputParser()
historial_conversacion: list[tuple[str, str]] = []


def traducir_a_ia(input_humano: str) -> tuple[str, str]:
    parsed = parsear_lenguaje_humano(input_humano)
    resultado_traduccion = cadena_traductor.invoke(
        {"input_humano": input_humano, "parsed_data": str(parsed)}
    )

    historial = (
        "\n".join(
            f"Humano: {entrada}\nAgente: {respuesta}"
            for entrada, respuesta in historial_conversacion
        )
        or "Sin historial previo."
    )
    respuesta_agente = cadena_agente.invoke(
        {"historial": historial, "traduccion": resultado_traduccion}
    )
    historial_conversacion.append((input_humano, respuesta_agente))
    return resultado_traduccion, respuesta_agente


if __name__ == "__main__":
    input_ejemplo = (
        "Quiero razonar sobre como programar un bot que traduce idiomas en Python."
    )
    traduccion, respuesta = traducir_a_ia(input_ejemplo)
    print("Traduccion a formato IA:\n", traduccion)
    print("\nRespuesta del agente:\n", respuesta)
