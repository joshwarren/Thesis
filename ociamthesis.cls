%
% ociamthesis v2.2
% By Keith A. Gillow <gillow@maths.ox.ac.uk>
% Version 1.0 released 26/11/1997
%-------------------------- identification ---------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ociamthesis}[2010/11/22 v2.2 OCIAM thesis class]
%-------------------------- initial code -----------------------
\def\logoversion{squarelogo}
\DeclareOption{beltcrest}{\def\logoversion{beltcrest}}
\DeclareOption{shieldcrest}{\def\logoversion{shieldcrest}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[a4paper]{report}
% As an alternative to the above could use next line for twosided output
%\LoadClass[a4paper,twoside,openright]{report}

\RequirePackage{graphicx} % needed for latest frontpage logo
\RequirePackage{ifthen}   % needed for option parsing for logo

\raggedbottom

%define the default submitted text
\newcommand{\submittedtext}{{A thesis submitted for the degree of}}

%
% DECLARATIONS
%
% These macros are used to declare arguments needed for the
% construction of the title page and other preamble.

% The year and term the thesis is submitted 
\def\degreedate#1{\gdef\@degreedate{#1}}
% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}
% The name of your Oxford college (e.g. Christ Church, Pembroke)
\def\college#1{\gdef\@college{#1}}


%
% Setup choosen crest/logo
%

\ifthenelse{\equal{\logoversion}{shieldcrest}}%
{
% Traditional Oxford shield crest
%Using latex metafont (Mathematical Institute system)
\font\crestfont=oxcrest40 scaled\magstep3
\def\logo{{\crestfont \char1}}
%For comlab system replace 1st line above with
%\font\crestfont=crest scaled\magstep3
}{}

\ifthenelse{\equal{\logoversion}{beltcrest}}%
{
% Newer Oxford Belt crest
%Using latex metafont (Mathematical Institute system)
\font\beltcrestfont=oxbeltcrest
\def\logo{{\beltcrestfont \char0}}
%For comlab system replace 1st line above with
%\font\beltcrestfont=newcrest
}{}

\ifthenelse{\equal{\logoversion}{squarelogo}}%
{
% Latest Logo, Square version (the default!)
% you need an oxlogo.eps or oxlogo.pdf file as appropriate
\def\logo{{\includegraphics[width=32mm]{oxlogo}}}
}{}

%
% Define text area of page and margin offsets
%
\setlength{\topmargin}{0.0in}
\setlength{\oddsidemargin}{0.33in}
\setlength{\evensidemargin}{-0.08in}
\setlength{\textheight}{9.0in}
\setlength{\textwidth}{6.0in}

%
% Environments
%

% This macro define an environment for front matter that is always 
% single column even in a double-column document.

\newenvironment{alwayssingle}{%
       \@restonecolfalse
       \if@twocolumn\@restonecoltrue\onecolumn
       \else\if@openright\cleardoublepage\else\clearpage\fi
       \fi}%
       {\if@restonecol\twocolumn
       \else\newpage\thispagestyle{empty}\fi}

%define title page layout
\renewcommand{\maketitle}{%
\begin{alwayssingle}
    \renewcommand{\footnotesize}{\small}
    \renewcommand{\footnoterule}{\relax}
    \thispagestyle{empty}
  \null\vfill
  \begin{center}
    { \Huge {\bfseries {\@title}} \par}
{\large \vspace*{25mm} {\logo \par} \vspace*{25mm}}
    {{\Large \@author} \par}
{\large \vspace*{1ex}
    {{\@college} \par}
\vspace*{1ex}
    {University of Oxford \par}
\vspace*{25mm}
    {{\submittedtext} \par}
\vspace*{1ex}
    {\it {\@degree} \par}
\vspace*{2ex}
    {\@degreedate}}
  \end{center}
  \null\vfill
\end{alwayssingle}}

% DEDICATION
%
% The dedication environment makes sure the dedication gets its
% own page and is set out in verse format.

\newenvironment{dedication}
{\begin{alwayssingle}
  \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\LARGE }
  \end{center}
  \vspace{0.5cm}
  \begin{verse}\begin{center}}
{\end{center}\end{verse}\end{alwayssingle}}


% ACKNOWLEDGEMENTS
%
% The acknowledgements environment puts a large, bold, centered 
% "Acknowledgements" label at the top of the page. The acknowledgements
% themselves appear in a quote environment, i.e. tabbed in at both sides, and
% on its own page.

\newenvironment{acknowledgements}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Acknowledgements}
\end{center}
\vspace{0.5cm}
\begin{quote}}
{\end{quote}\end{alwayssingle}}

% The acknowledgementslong environment puts a large, bold, centered 
% "Acknowledgements" label at the top of the page. The acknowledgement itself 
% does not appears in a quote environment so you can get more in.

\newenvironment{acknowledgementslong}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Acknowledgements}
\end{center}
\vspace{0.5cm}}
{\end{alwayssingle}}

% STATEMENT OF ORIGINALITY (AS SUGGESTED BY GSW)
%
% The originality environment puts a large, bold, centered 
% "Statement of originality" label at the top of the page. The statement 
% of originality itself appears in a quote environment, i.e. tabbed in at 
% both sides, and on its own page.

\newenvironment{originality}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Statement of Originality}
\end{center}
\vspace{0.5cm}
\begin{quote}}
{\end{quote}\end{alwayssingle}}

% The originalitylong environment puts a large, bold, centered 
% "Statement of originality" label at the top of the page. The statement 
% of originality itself does not appears in a quote environment so you can 
% get more in.

\newenvironment{originalitylong}
{\begin{alwayssingle} \thispagestyle{empty}
\begin{center}
\vspace*{1.5cm}
{\Large \bfseries Statement of Originality}
\end{center}
\vspace{0.5cm}}
{\end{alwayssingle}}


%ABSTRACT
%
%The abstract environment puts a large, bold, centered "Abstract" label at
%the top of the page. The abstract itself appears in a quote environment,
%i.e. tabbed in at both sides, and on its own page.

\renewenvironment{abstract} {\begin{alwayssingle} \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\Large \bfseries  Abstract}
  \end{center}
  \vspace{0.5cm}
   \begin{quote}}
{\end{quote}\end{alwayssingle}}

%The abstractlong environment puts a large, bold, centered "Abstract" label at
%the top of the page. The abstract itself does not appears in a quote
%environment so you can get more in.

\newenvironment{abstractlong} {\begin{alwayssingle} \thispagestyle{empty}
  \begin{center}
  \vspace*{1.5cm}
  {\Large \bfseries  Abstract}
  \end{center}
  \vspace{0.5cm}}
{\end{alwayssingle}}

%The abstractseparate environment is for running of a page with the abstract
%on including title and author etc as required to be handed in separately

\newenvironment{abstractseparate} {\begin{alwayssingle} \thispagestyle{empty}
  \vspace*{-1in}
 \begin{center}
    { \Large {\bfseries {\@title}} \par}
    {{\large \vspace*{1ex} \@author} \par}
{\large \vspace*{1ex}
    {{\@college} \par}
    {University of Oxford \par}
\vspace*{1ex}
    {{\it \submittedtext} \par}
    {\it {\@degree} \par}
\vspace*{2ex}
    {\@degreedate}}
  \end{center}}
{\end{alwayssingle}}

%ROMANPAGES
%
% The romanpages environment set the page numbering to lowercase roman one
% for the contents and figures lists. It also resets
% page-numbering for the remainder of the dissertation (arabic, starting at 1).

\newenvironment{romanpages}
{\cleardoublepage\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}




% Standard journal abbreviations
% Mostly as used by ADS, with a few additions for journals where MNRAS does not
% follow normal IAU style.

\newcommand\aap{A\&A}                % Astronomy and Astrophysics
\let\astap=\aap                          % alternative shortcut
\newcommand\aapr{A\&ARv}             % Astronomy and Astrophysics Review (the)
\newcommand\aaps{A\&AS}              % Astronomy and Astrophysics Supplement Series
\newcommand\actaa{Acta Astron.}      % Acta Astronomica
\newcommand\afz{Afz}                 % Astrofizika
\newcommand\aj{AJ}                   % Astronomical Journal (the)
\newcommand\ao{Appl. Opt.}           % Applied Optics
\let\applopt=\ao                         % alternative shortcut
\newcommand\aplett{Astrophys.~Lett.} % Astrophysics Letters
\newcommand\apj{ApJ}                 % Astrophysical Journal
\newcommand\apjl{ApJ}                % Astrophysical Journal, Letters
\let\apjlett=\apjl                       % alternative shortcut
\newcommand\apjs{ApJS}               % Astrophysical Journal, Supplement
\let\apjsupp=\apjs                       % alternative shortcut
% The following journal does not appear to exist! Disabled.
%\newcommand\apspr{Astrophys.~Space~Phys.~Res.} % Astrophysics Space Physics Research
\newcommand\apss{Ap\&SS}             % Astrophysics and Space Science
\newcommand\araa{ARA\&A}             % Annual Review of Astronomy and Astrophysics
\newcommand\arep{Astron. Rep.}       % Astronomy Reports
\newcommand\aspc{ASP Conf. Ser.}     % ASP Conference Series
\newcommand\azh{Azh}                 % Astronomicheskii Zhurnal
\newcommand\baas{BAAS}               % Bulletin of the American Astronomical Society
\newcommand\bac{Bull. Astron. Inst. Czechoslovakia} % Bulletin of the Astronomical Institutes of Czechoslovakia 
\newcommand\bain{Bull. Astron. Inst. Netherlands} % Bulletin Astronomical Institute of the Netherlands
\newcommand\caa{Chinese Astron. Astrophys.} % Chinese Astronomy and Astrophysics
\newcommand\cjaa{Chinese J.~Astron. Astrophys.} % Chinese Journal of Astronomy and Astrophysics
\newcommand\fcp{Fundamentals Cosmic Phys.}  % Fundamentals of Cosmic Physics
\newcommand\gca{Geochimica Cosmochimica Acta}   % Geochimica Cosmochimica Acta
\newcommand\grl{Geophys. Res. Lett.} % Geophysics Research Letters
\newcommand\iaucirc{IAU~Circ.}       % IAU Cirulars
\newcommand\icarus{Icarus}           % Icarus
\newcommand\japa{J.~Astrophys. Astron.} % Journal of Astrophysics and Astronomy
\newcommand\jcap{J.~Cosmology Astropart. Phys.} % Journal of Cosmology and Astroparticle Physics
\newcommand\jcp{J.~Chem.~Phys.}      % Journal of Chemical Physics
\newcommand\jgr{J.~Geophys.~Res.}    % Journal of Geophysics Research
\newcommand\jqsrt{J.~Quant. Spectrosc. Radiative Transfer} % Journal of Quantitiative Spectroscopy and Radiative Transfer
\newcommand\jrasc{J.~R.~Astron. Soc. Canada} % Journal of the RAS of Canada
\newcommand\memras{Mem.~RAS}         % Memoirs of the RAS
\newcommand\memsai{Mem. Soc. Astron. Italiana} % Memoire della Societa Astronomica Italiana
\newcommand\mnassa{MNASSA}           % Monthly Notes of the Astronomical Society of Southern Africa
\newcommand\mnras{MNRAS}             % Monthly Notices of the Royal Astronomical Society
\newcommand\na{New~Astron.}          % New Astronomy
\newcommand\nar{New~Astron.~Rev.}    % New Astronomy Review
\newcommand\nat{Nature}              % Nature
\newcommand\nphysa{Nuclear Phys.~A}  % Nuclear Physics A
\newcommand\pra{Phys. Rev.~A}        % Physical Review A: General Physics
\newcommand\prb{Phys. Rev.~B}        % Physical Review B: Solid State
\newcommand\prc{Phys. Rev.~C}        % Physical Review C
\newcommand\prd{Phys. Rev.~D}        % Physical Review D
\newcommand\pre{Phys. Rev.~E}        % Physical Review E
\newcommand\prl{Phys. Rev.~Lett.}    % Physical Review Letters
\newcommand\pasa{Publ. Astron. Soc. Australia}  % Publications of the Astronomical Society of Australia
\newcommand\pasp{PASP}               % Publications of the Astronomical Society of the Pacific
\newcommand\pasj{PASJ}               % Publications of the Astronomical Society of Japan
\newcommand\physrep{Phys.~Rep.}      % Physics Reports
\newcommand\physscr{Phys.~Scr.}      % Physica Scripta
\newcommand\planss{Planet. Space~Sci.} % Planetary Space Science
\newcommand\procspie{Proc.~SPIE}     % Proceedings of the Society of Photo-Optical Instrumentation Engineers
\newcommand\rmxaa{Rev. Mex. Astron. Astrofis.} % Revista Mexicana de Astronomia y Astrofisica
\newcommand\qjras{QJRAS}             % Quarterly Journal of the RAS
\newcommand\sci{Science}             % Science
\newcommand\skytel{Sky \& Telesc.}   % Sky and Telescope
\newcommand\solphys{Sol.~Phys.}      % Solar Physics
\newcommand\sovast{Soviet~Ast.}      % Soviet Astronomy (aka Astronomy Reports)
\newcommand\ssr{Space Sci. Rev.}     % Space Science Reviews
\newcommand\zap{Z.~Astrophys.}       % Zeitschrift fuer Astrophysik