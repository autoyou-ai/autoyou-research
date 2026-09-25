"""Build the revised paper without latexmk/Perl; requires pdflatex and bibtex."""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
STEM = "right-sized-edge"


def main():
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        ["bibtex", STEM],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
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
    log = (PAPER / f"{STEM}.log").read_text(errors="replace")
    problems = re.findall(r"^.*(?:undefined|Overfull|LaTeX Error|multiply defined).*$", log, re.M)
    if problems:
        raise SystemExit("Build QA failed:\n" + "\n".join(problems))
    print("PASS: no undefined references, overfull boxes, duplicate labels, or LaTeX errors.")
    print(PAPER / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
