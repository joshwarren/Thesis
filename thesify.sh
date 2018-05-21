#!/bin/bash
sed -i -e 's/Mon. Not. R. Astron. Soc/\\mnras/g' refs.bib
sed -i -e 's/Monthly Notices of the Royal Astronomical Society/\\mnras/g' refs.bib
sed -i -e 's/The Astrophysical Journal/\\apj/g' refs.bib
sed -i -e 's/{Astrophysical Journal}/{\\apj}/g' refs.bib
sed -i -e 's/The Astronomical Journal/\\aj/g' refs.bib
sed -i -e 's/{Astronomical Journal}/{\\aj}/g' refs.bib
sed -i -e 's/Annual Review of Astronomy and Astrophysics/\\araa/g' refs.bib
sed -i -e 's/Annu. Rev. Astron. Astrophys./\\araa/g' refs.bib
sed -i -e 's/Publications of the Astronomical Society of the Pacific/\\pasp/g' refs.bib
sed -i -e 's/The Astronomy and Astrophysics Review/\\aapr/g' refs.bib
sed -i -e 's/Astronomy and Astrophysics/\\aap/g' refs.bib
sed -i -e 's/Astronomy {\\&} Astrophysics/\\aap/g' refs.bib
sed -i -e 's/Publications of the Astronomical Society of Australia/\\pasa/g' refs.bib
sed -i -e 's/Proceedings of the Astronomical Society of Australia/\\pasa/g' refs.bib
sed -i -e 's/Physical Review Letters/\\prl/g' refs.bib
sed -i -e 's/Bulletin of the American Astronomical Society/\\baas/g' refs.bib
sed -i -e 's/journal = {Nature}/journal = {\\nat}/g' refs.bib
sed -i -e 's/journal = {Science.*}/journal = {\\sci}/g' refs.bib
sed -i -e 's/Physical Review D/\\prd/g' refs.bib
sed -i -e 's/New Astronomy Reviews/\\nar/g' refs.bib
sed -i -e 's/Journal of Cosmology and Astroparticle Physics/\\jcap/g' refs.bib

# cat refs.bib | tr '\n' '\f' | sed -e 's/booktitle = {Proc. SPIE},\fdoi = {10.1117\/12.856027}/booktitle = {Proc. SPIE},\nseries = {III},\ndoi = {10.1117\/12.856027}/'  | tr '\f' '\n' > refs.bib


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