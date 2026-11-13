"""Compatibility entry point for the shared recorded hand-eye workflow.

No arguments: capture with persistent storage. Use `solve` separately on a session.
"""
import sys

from handeye.__main__ import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or arguments[0].startswith("-"):
        arguments = ["capture", *arguments]
    if arguments[0] == "capture" and "--mode" not in arguments:
        arguments += ["--mode", "eye_to_hand"]
    main(arguments)
