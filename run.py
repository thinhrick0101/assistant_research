import streamlit.web.cli as stcli
import sys
import os


def main():
    sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py")]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
