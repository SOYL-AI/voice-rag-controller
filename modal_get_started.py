import modal


app = modal.App("voice-rag-get-started")


@app.function()
def square(x: int) -> int:
    print("This code is running on a remote Modal worker!")
    return x**2


@app.local_entrypoint()
def main() -> None:
    print("the square is", square.remote(42))
