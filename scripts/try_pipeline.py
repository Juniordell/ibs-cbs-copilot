import sys, asyncio, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.copilot.pipeline import answer_question


async def main():
    result = await answer_question("Qual a alíquota do IBS?", k=5)
    a = result.generation.answer
    print(f"\nANSWER ({a.confidence}):\n{a.answer}\n")
    print("CITATIONS:")
    for c in a.citations:
        print(f"  · {c.article} · {c.source}")
        print(f'    "{c.quote}"')
    print(f"\nTokens: in={result.generation.input_tokens} out={result.generation.output_tokens}")


asyncio.run(main())