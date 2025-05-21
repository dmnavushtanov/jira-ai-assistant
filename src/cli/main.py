import typer

app = typer.Typer(help="Jira AI Assistant CLI")

@app.command()
def hello():
    """Simple placeholder command."""
    typer.echo("Hello from Jira AI Assistant!")

if __name__ == "__main__":
    app()
