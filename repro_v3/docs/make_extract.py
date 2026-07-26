"""Regenerate the standalone numerical-section extract from main.tex."""
import io

src = io.open("main.tex", encoding="utf-8").read().split("\n")

def find(prefix):
    for k, l in enumerate(src):
        if l.startswith(prefix):
            return k
    raise SystemExit("not found: " + prefix)

bd = find("\\begin{document}")
i = find("\\section{Computational Studies}")
j = find("\\section{Conclusion")

header = (
    "\\begin{document}\n"
    "{\\Large\\bfseries Compressed Traffic Assignment --- Numerical Section}\\par\n"
    "\\vspace{3pt}\n"
    "{\\small\\textsf{\\docstamp}}\\par\n"
    "\\vspace{3pt}\n"
    "{\\small Standalone extract of the rewritten computational study. Body text is black;\n"
    "\\textcolor{red}{red marks notes to co-authors} only, to be removed before submission.\n"
    "Every number traces to a CSV in \\texttt{repro\\_v3/results}.}\n"
    "\\vspace{8pt}\\hrule\\vspace{8pt}\n"
)

doc = "\n".join(src[:bd]) + "\n" + header + "\n".join(src[i:j]) + "\n\\end{document}\n"
io.open("numerical_section.tex", "w", encoding="utf-8", newline="\n").write(doc)
print("preamble lines %d, section lines %d-%d" % (bd, i + 1, j))
