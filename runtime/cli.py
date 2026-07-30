"""
=========================================================
QAIR (Quantom AI Runtime)

File:
    runtime/cli.py

Purpose:
    Interactive command-line interface for QAIR.

Author:
    TIOTAIROBOTIX
=========================================================
"""

import os

from runtime.inference.engine import LlamaEngine
from runtime.chat.session import ChatSession


# ---------------------------------------------------------
# Banner
# ---------------------------------------------------------

def banner():

    print("\n" + "=" * 60)
    print("        QAIR (Quantom AI Runtime)")
    print("        TIOTAIROBOTIX")
    print("=" * 60)

    print("\nLocal AI Runtime")
    print("Type 'help' for commands.")
    print("Type 'exit' to quit.\n")


# ---------------------------------------------------------
# Help
# ---------------------------------------------------------

def help_menu():

    print("""
Available Commands
------------------

help
    Show this menu.

info
    Show runtime information.

history
    Show conversation history.

reset
    Clear current conversation.

clear
    Clear the screen.

exit
quit
    Exit QAIR.
""")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    banner()

    print("Loading Engine...\n")

    engine = LlamaEngine()

    session = ChatSession(engine)

    print("QAIR Ready.\n")

    while True:

        try:

            user = input("QAIR > ").strip()

            if not user:
                continue

            command = user.lower()

            if command in ("exit", "quit"):

                print("\nGoodbye.\n")
                break

            elif command == "help":

                help_menu()
                continue

            elif command == "info":

                print()
                print(engine.info())
                print()
                continue

            elif command == "history":

                print()

                history = session.history()

                if not history:
                    print("No conversation yet.")

                else:

                    for msg in history:

                        print(
                            f"{msg['role'].capitalize()}: {msg['content']}"
                        )

                print()
                continue

            elif command == "reset":

                session.clear()

                print("\nConversation cleared.\n")

                continue

            elif command == "clear":

                os.system("clear")

                banner()

                continue

            response = session.ask(user)

            print()

            print(response)

            print()

        except KeyboardInterrupt:

            print("\n\nInterrupted.\n")

            break

        except Exception as e:

            print(f"\nRuntime Error:\n{e}\n")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
