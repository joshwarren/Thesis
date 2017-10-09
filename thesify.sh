#!/bin/bash

pdflatex -interaction=batchmode thesis.tex
bibtex thesis
pdflatex -interaction=batchmode thesis.tex 
pdflatex -interaction=batchmode thesis.tex 

if [ `whoami` = 'warrenj' ]
then
	open thesis.pdf
else
	cygstart thesis.pdf # opens thesis.pdf with cygwin
fi