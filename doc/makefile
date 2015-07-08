LTX=pdflatex
BIB=bibtex
VIEW=evince

all:
	$(LTX) -shell-escape -draftmode paper
	$(BIB) paper
	$(LTX) -shell-escape -draftmode paper
	$(LTX) -shell-escape paper

view: all
	$(VIEW) paper.pdf

clean:
	$(RM) *.aux *.lof *.out *.toc *.log *.lot *.blg *.bbl

