#!/bin/bash

pdflatex -interaction=batchmode thesis.tex
bibtex thesis
pdflatex -interaction=batchmode thesis.tex 
pdflatex -interaction=batchmode thesis.tex 
cygstart thesis.pdf # opens thesis.pdf with cygwin