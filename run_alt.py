#!/usr/bin/env python
import streamlit.web.cli as stcli
import sys
import os


def main():
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "app_alt.py"),
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
