import os

from documind.app import create_app

app = create_app(os.environ.get("DOCUMIND_DB", "documind.db"))

if __name__ == "__main__":
    app.run(port=5000, debug=False)
