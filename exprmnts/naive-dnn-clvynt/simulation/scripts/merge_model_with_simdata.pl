#! /usr/bin/env perl

use strict;

my $PRED_FILE_GVKEY_IDX    = 0;
my $PRED_FILE_DATE_IDX     = 1;
my $PRED_FILE_POS_PROB_IDX = 2;

my $SIMDAT_FILE_DATE_IDX   = 0;
my $SIMDAT_FILE_GVKEY_IDX  = 1;
my $SIMDAT_FILE_TERM_IDX   = 11;

main();

sub main {

    $| = 1;

    my $datafile = $ARGV[0] || die "First cmd arg must be sim data file.";
    my $probfile = $ARGV[1] || die "Second cmd arg must be predictions file.";

    my %probs = ();

    open(F1,"< $probfile");

    while(<F1>) {

	chomp;
	my @fields = split(' ',$_);

	next if $fields[0] eq 'gvkey'; # skip header

	my $gvkey = $fields[$PRED_FILE_GVKEY_IDX];
	my $date  = $fields[$PRED_FILE_DATE_IDX];
	my $prob  = $fields[$PRED_FILE_POS_PROB_IDX];
	$probs{"$gvkey$date"} = $prob;

    }

    close(F1);

    open(F2,"< $datafile");

    while(<F2>) {

	chomp;
	my @fields = split(' ',$_);

	my $gvkey = $fields[$SIMDAT_FILE_GVKEY_IDX];
	my $date  = $fields[$SIMDAT_FILE_DATE_IDX];
	my $prob = '0.50';

	if (exists($probs{"$gvkey$date"})) {
	    $prob = $probs{"$gvkey$date"};
	}

	print join(' ',@fields[0..$SIMDAT_FILE_TERM_IDX]);

	print " $prob\n";

    }

    close(F2);

}
