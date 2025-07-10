from app.config import Config
from app import create_app, init_db

def main():
    # Crea y configura la aplicación Flask
    app = create_app()
    init_db()

    port = Config.PORT
    print(port)

    debug_mode = Config.DEBUG
    app.run(debug=debug_mode, port=port)

if __name__ == '__main__':
    main()