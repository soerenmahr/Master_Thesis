# latexmk configuration
$pdf_mode = 1;            # PDF via pdflatex
$out_dir  = 'build';      # PDF und Hilfsdateien landen hier
$bibtex_use = 2;          # Biber/BibTeX automatisch ausführen
$clean_ext = 'bbl run.xml';  # zusätzlich mit -c/-C entfernen

# --- Glossary support (makeglossaries) -----------------------
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
sub run_makeglossaries {
    my ($base_name, $path) = fileparse( $_[0] );
    pushd $path;
    my $ret = system( 'makeglossaries', $base_name );
    popd;
    return $ret;
}
$clean_ext .= ' glo gls glg acn acr alg ist xdy';