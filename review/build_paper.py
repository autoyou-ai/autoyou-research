"""Build the revised paper without latexmk/Perl; requires pdflatex and bibtex."""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def main():
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for command in commands:
        if not shutil.which(command[0]):
            raise SystemExit(f"Required executable missing: {command[0]}")
        completed = subprocess.run(command, cwd=PAPER, capture_output=True, text=True)
        if completed.returncode:
            print(completed.stdout)
            print(completed.stderr)
            raise SystemExit(completed.returncode)
        print(f"PASS: {' '.join(command)}")
    log = (PAPER / "main.log").read_text(errors="replace")
    problems = re.findall(r"^.*(?:undefined|Overfull|LaTeX Error|multiply defined).*$", log, re.M)
    if problems:
        raise SystemExit("Build QA failed:\n" + "\n".join(problems))
    print("PASS: no undefined references, overfull boxes, duplicate labels, or LaTeX errors.")
    print(PAPER / "main.pdf")


if __name__ == "__main__":
    main()
