import os, json
from textual_serve.server import Server
# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from typing import Tuple
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import Container


#texto colorido
from rich.console import Console

console = Console()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

if not os.environ["OPENAI_API_KEY"]:
    raise ValueError("OPENAI_API_KEY não definido no ambiente.")

SYSTEM_PROMPT = SystemMessagePromptTemplate.from_template(
    """Você é um analisador de emoções
    Receberá uma descrição de como o usuario se sente. Responda ESTRITAMENTE em JSON:

    
    (
    "explanation": "<explicação curta, clara, em português, no máximo 120 palavras do motivo da analise>",
    "emocao": "<triste ou feliz ou naosei, só pode retornar esses 3 valores>",
    "escala": "<valor numerico de 0 a 10, onde 0 é o menos e 10 é o mais representando o nivel de felicidade>"
    )

    Regras:
    - Não invente opções inexistentes.
    - Nunca inclua texto fora do JSON.
 
    """
)


USER_PROMPT = HumanMessagePromptTemplate.from_template(
    """{query}""",
    input_variables=["query"]
)
USER_PROMPT.format(query = "query")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(temperature=0, model=DEFAULT_MODEL)
    
prompt = ChatPromptTemplate.from_messages([SYSTEM_PROMPT, USER_PROMPT])

chain= (
    {"query": lambda x: x["query"]} #inputs
    | prompt # o que está em cima entra no que ta em baixo
    | llm#meu modelo com temperatura alta
    | {"saida": lambda x: x.content} #pega a saida
)


def queryAI(user_query: str) -> Tuple[str, str]: 
    return json.loads(chain.invoke({"query": user_query})["saida"])



class EmoApp(App):
    def on_mount(self) -> None:
        self.theme = "nord"

    def compose(self) -> ComposeResult:
        # Cabeçalho e rodapé
        yield Header()
        yield Container(
            Input(placeholder="Digite como você está se sentindo...", id="entrada"),
            Static("", id="resultado")  # aqui mostramos a resposta
        )
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        texto = event.value
        resposta = queryAI(texto)
        explicacao = resposta["explanation"]
        emocao = resposta["emocao"]
        nivel = resposta["escala"]

        # mostra bonitinho no widget Static
        resultado = self.query_one("#resultado", Static)
        resultado.update(f"[b]{emocao.upper()}[/b] [pink] {nivel} [/pink]  — {explicacao}")



if __name__ == "__main__":
    app = EmoApp()
    app.run()

