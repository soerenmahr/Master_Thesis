# Master Thesis - LaTeX Template

## Structure
```
main.tex                     master file, assembles everything
config/
    preamble.tex             packages and global settings
    commands.tex             custom macros
frontmatter/
    titlepage.tex
    abstract.tex             abstract + Zusammenfassung
    acknowledgements.tex
    declaration.tex          declaration of authorship
chapters/
    01_introduction.tex
    02_theory.tex
    03_methods.tex
    04_results.tex
    05_conclusion.tex
appendix/
    appendix.tex
bibliography/
    references.bib
figures/                     figure files go here
```

## Compilation
The template uses BibLaTeX with the Biber backend. A `.latexmkrc` is
included that puts all output (PDF + auxiliary files) into a `build/`
folder, so the source tree stays clean.

Simplest (recommended):
```
latexmk
```
The PDF is then at `build/main.pdf`.

Without the .latexmkrc, or to override the folder:
```
latexmk -pdf -outdir=build main.tex
```

Manual sequence (writes into the current folder):
```
pdflatex main
biber main
pdflatex main
pdflatex main
```

Clean the build folder:
```
latexmk -C
```

## Language
The document language is set once via the class option in `main.tex`
(`english` -> switch to `ngerman` for a German thesis). `babel` and
`cleveref` follow this option automatically.

## Notes
- `\todo{...}` marks placeholders in red; remove `lipsum` and all `\todo`
  commands before submission.
- Bibliography style is `numeric-comp`. If `biblatex-phys` is installed,
  change `style=numeric-comp` to `style=phys` in `config/preamble.tex`.
- Adjust margins, examiner names, and the declaration wording to your
  institution's requirements.
```
