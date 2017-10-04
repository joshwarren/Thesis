#!/bin/bash

pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
cygstart thesis.pdf # opens thesis.pdf with cygwin