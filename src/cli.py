import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from rag_agent import F1RAGAgent

load_dotenv()

console = Console()

def run_cli():
    console.print(Panel.fit(
        "[bold red]2026 F1 Spec RAG Agent[/bold red]\n"
        "Ask technical questions about the 2026 FIA Technical Regulations.",
        title="Welcome", border_style="red"
    ))
    
    # Check if vector db exists
    if not os.path.exists("vector_db"):
        console.print("[yellow]Warning: Vector database not found. Please run 'python src/ingestion.py' first.[/yellow]")
        return

    try:
        agent = F1RAGAgent()
    except Exception as e:
        console.print(f"[red]Error initializing RAG agent: {e}[/red]")
        return

    while True:
        try:
            question = console.input("\n[bold cyan]Ask a question (or 'exit' to quit): [/bold cyan]")
            
            if question.lower() in ["exit", "quit", "q"]:
                console.print("[green]Goodbye![/green]")
                break
            
            if not question.strip():
                continue

            with console.status("[bold green]Analyzing regulations and generating answer...[/bold green]"):
                response = agent.query(question)
            
            console.print("\n", Rule(title="[bold yellow]Answer[/bold yellow]", style="yellow"))
            console.print(Markdown(response["answer"]))
            
            console.print("\n", Rule(title="[bold blue]Source Context[/bold blue]", style="blue"))
            for i, doc in enumerate(response["context"][:3]): # Show top 3 sources
                page_num = doc.metadata.get('page', 'N/A')
                if isinstance(page_num, int):
                    page_num += 1
                console.print(f"[bold blue]Source {i+1} (Page {page_num}):[/bold blue]")
                console.print(f"[italic]{doc.page_content[:300]}...[/italic]")
                console.print("-" * 20)

            if response.get("skill_active"):
                console.print("\n", Rule(title="[bold magenta]Live Web Sources[/bold magenta]", style="magenta"))
                if response.get("skill_error"):
                    console.print(f"[yellow]{response['skill_error']}[/yellow]")
                elif not response.get("web_sources"):
                    console.print("[yellow]No live web snippets were returned for this question.[/yellow]")
                else:
                    for i, source in enumerate(response["web_sources"], start=1):
                        console.print(f"[bold magenta]{i}. {source['title']}[/bold magenta]")
                        console.print(source["url"])
                        console.print(f"[italic]{source['snippet']}[/italic]")
                        console.print("-" * 20)

        except KeyboardInterrupt:
            console.print("\n[green]Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[red]An error occurred: {e}[/red]")

if __name__ == "__main__":
    run_cli()
