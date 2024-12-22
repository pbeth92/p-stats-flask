from app.config import Config
from app import create_app, init_db

def main():
    app = create_app()
    init_db()

    port = Config.PORT
    debug_mode = Config.DEBUG

    app.run(debug=debug_mode, port=port)

if __name__ == '__main__':
    main()