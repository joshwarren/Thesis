#!/bin/bash

function sort_latex {
	current_dir=$(pwd)
	cd /home/HOME/Documents/thesis_changes
	# rescue git folder from flattening
	mv -f .git ../

	# rename vimos and muse figures as they have the same name
	for instr in vimos muse
	do
		for f in $( ls */$instr/* )
		do
			mv $f $( echo $f | sed 's/\//_/g' | sed 's/_/\//' )
		done

		rm -r */$instr
	done
	# flatten
	find . -mindepth 2 -type f -exec mv -i -f '{}' . ';'
	mv -f ../.git ./

	# remove references to folders
	for f in $( ls *.tex )
	do
		sed 's/introduction\///g' $f | sed 's/chapter2\///g' \
			| sed 's/chapter4\///g' | sed 's/chapter5\///g' \
			| sed 's/conclusions\///g' | sed 's/appendix\/appendix2\///g' \
			| sed 's/vimos\//vimos_/g' | sed 's/muse\//muse_/g' \
			| sed 's/acknowledgements\///g' | sed 's/abstract\///g'> temp
		mv temp $f
	done

	# flatten latex documents
	for f in abstract origionality acknowledgements introduction chapter2 \
	chapter4 chapter5 conclusions appendix2
	do
		sed -i "s/\\include{"$f"}/$(sed -e 's/[\&/]/\\&/g' \
			-e 's/$/\\n/' $f.tex | tr -d '\n')/" thesis.tex
		rm $f.*
	done
	# Removing unnecessary double \ from above sed command
	sed -i 's/\\\\b/\\b/g' thesis.tex
	sed -i 's/\\\\c/\\c/g' thesis.tex

	# remove uneeded files
	rm -r abstract
	rm -r acknowledgements
	rm -r introduction
	rm -r chapter2
	rm -r chapter4
	rm -r chapter5
	rm -r conclusions
	rm -r appendix*	
	rm *.aux
	rm *.bbl
	rm *.blg
	rm *.dvi
	rm *.lof
	rm *.log
	rm *.out
	rm *.pdf
	rm *.toc 
	rm *.sh
	rm *.py

	cd $current_dir
}

function update {
	echo "Updating copies of old and most recent repos"

	current_dir=$(pwd)
	cd /home/HOME/Documents/thesis
	cp -r * /home/HOME/Documents/thesis_changes_new/

	git stash
	git checkout 44c3183a0115766126279a6a011dd4c437e8d7ea
	cp -r * /home/HOME/Documents/thesis_changes_old/
	git checkout master
	git stash pop

	cd $current_dir
}

function run_all_plots {
	echo 'This function needs to called from Ubuntu on Windows utility'

	current_dir=$(pwd)
	cd /home/joshwarren/Documents/thesis

	cd chapter2
	python produce_plots2.py

	cd ../chapter4
	python produce_plots2.py
	python plot_resolved_lambda_R.py

	cd ../chapter5
	python produce_plots2.py

	cd /home/joshwarren/MUSE_project/analysis
	python compare_atlas3d.py thesis
	python BPT_muse.py

	cd /home/joshwarren/VIMOS_project/analysis
	python BPT.py

	cd $current_dir
}

function correct_latex {
	echo "Correcting the LaTeX produced by latexdiff"

	changes_file=/home/HOME/Documents/thesis_changes/changes.tex
	dos2unix $changes_file
	cat $changes_file | tr '\r' '\n' > tmp
	mv tmp $changes_file

	# remove indents
	cat $changes_file | sed "s/^[ \t]*//g" > tmp
	mv tmp $changes_file

	# Remove all comments (begining of line first (remove preceding and 
	# 	following newline), then inline)
	# cat $changes_file | tr '\n' '~' | sed 's/~mxc/qwerty/g' \
	# 	| sed -r 's/(\b|[^\\])\%[^~]*~/\1/g' | sed 's/~\%[^~]*~//g' \
	# 	| sed -r 's/(\b|[^\\])\%[^~]*~/\1/g' |tr '~' '\n' \
	# 	| sed 's/qwerty/~mxc/g' > tmp
	# mv tmp $changes_file
	cat $changes_file | tr '\n' '~' | sed 's/~mxc/qwerty/g' \
		| sed -r 's/(\b|[^\\])\%[^~]*/\1/g' | tr '~' '\n' \
		| sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	# Max of 1 empty line
	cat $changes_file | tr '\n' '~' | sed 's/~~~*/~~/g' | sed 's/~mxc/qwerty/g' \
		| tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	# Remove uncessary newlines
	cat $changes_file | tr '\n' '~' | sed 's/~mxc/qwerty/g' \
		| sed 's/\({[^}^~]*\)~~*/\1/g' | sed 's/~~*}/}/g' | tr '~' '\n' \
		| sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/\. *~*\}/\.\}/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/~*\}\\DIF/\}\\DIF/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/mbox{~*/mbox{/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/hspace{0}~*/hspace{0pt} /g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/}~*\\DIFaddend/}\\DIFaddend/g' | \
		sed 's/}~*\\DIFdelend/}\\DIFdelend/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	# Re-add necessary newlines
	cat $changes_file | tr '\n' '~' | sed 's/end *\\end/end~\\end/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	cat $changes_file | tr '\n' '~' | sed 's/  *\\subs/~~\\subs/g' | \
		sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	# Correcting error in old version
	sed -i 's/Con topoulos1956/Contopoulos1956/g' $changes_file

	# Remove empty del and add enviroments
	cat $changes_file | tr '\n' '~' | sed 's/\\DIFdelbegin *~* *\\DIFdelend *//g' \
		| sed 's/\\DIFaddbegin *~* *\\DIFaddend *//g' \
		| sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	mv tmp $changes_file

	# Correct output of latex diff - change newline to ~
	
	# cat $changes_file | tr '\n' '~' | sed 's/\. *~*\}/\.\}/g' \
	# 	| sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	# mv tmp $changes_file
	# cat $changes_file | tr '\n' '~' | sed 's/\.~\%.*~~\}/\.\}/g' \
	# 	| sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	# mv tmp $changes_file

	# cat $changes_file | tr '\n' '~' | sed 's/\}~*\\DIF/\}\\DIF/g' | \
	# 	sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	# mv tmp $changes_file

	# To include Windows style newlines 
	# cat $changes_file | sed -r ':a;N;$!ba;s/\.\r\n\r\}/\.\}/g'  > tmp
	# mv tmp $changes_file
	# cat $changes_file | sed -r ':a;N;$!ba;s/\. \r\n\r\}/\.\}/g' > tmp
	# mv tmp $changes_file
	# cat $changes_file | sed -r ':a;N;$!ba;s/\.\n\%.*\r\n\r\}/\.\}/g' > tmp
	# mv tmp $changes_file

	

	# cat $changes_file | tr '\n' '~' | sed 's/~\}/\}/g' | sed 's/\}~[^~]/\}/' \
	# 	| sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp
	# mv tmp $changes_file
	# cat $changes_file | sed ':a;N;$!ba;s/\.\r\n\r\}/\.\}/g'  > tmp
	# mv tmp $changes_file


	# cat $changes_file | tr '\n' '~' | sed 's/{~*/\{/g' \
	# 	| sed 's/~mxc/qwerty/g' | tr '\~' '\n' | sed 's/qwerty/~mxc/g' > tmp 
		# | sed 's/\}~\\DIF/\}\\DIF/g' 
		# | sed 's/~*\}/\}/g'
	# diff tmp $changes_file
	# mv tmp $changes_file
}

function thesify {
	echo "Producing PDF document changes.pdf"

	current_dir=$(pwd)
	cd /home/HOME/Documents/thesis_changes

	pdflatex -interaction=batchmode changes.tex
	bibtex changes
	pdflatex -interaction=batchmode changes.tex
	pdflatex -interaction=batchmode changes.tex

	if [ `whoami` = 'warrenj' ]
	then
		open changes.pdf
	else
		cygstart changes.pdf # opens thesis.pdf with cygwin
	fi

	cd $current_dir
}

function create_changes {
	echo "Running latexdiff utility"

	current_dir=$(pwd)
	cd /home/HOME/Documents/thesis_changes

	rm -rf /home/HOME/Documents/thesis_changes/*

	## --------------------=========== Origional ==========-------------- ##
	cp -r /home/HOME/Documents/thesis_changes_old/* \
		/home/HOME/Documents/thesis_changes/
	sort_latex
	mv thesis.tex thesis.old

	## --------------------=========== Changed ==========-------------- ##
	cp -r /home/HOME/Documents/thesis_changes_new/* \
		/home/HOME/Documents/thesis_changes/
	sort_latex
	mv thesis.tex thesis.new

	## --------------------=========== Comparison ==========-------------- ##
	cp /home/HOME/Documents/thesis/refs.bib /home/HOME/Documents/thesis_changes/
	latexdiff thesis.old thesis.new > changes.tex
	cd $current_dir
}


function run {
	update
	create_changes
	correct_latex
	thesify
}




# Taken from https://stackoverflow.com/questions/8818119/
# 	how-can-i-run-a-function-from-a-script-in-command-line
# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi


