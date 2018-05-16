#!/usr/local/bin/perl -w

use strict;

$| = 1;

my @lines = ();

my $tic_file = $ARGV[0];
my $min_date = $ARGV[1] || '190001';

my %GVKEY_TO_TIC = ( );

open(FP,"< $tic_file") || die "Could not open file $tic_file.";

while(<FP>) {
    chomp;
    my ($gvkey,$tic) = split ' ',$_;
    die unless $gvkey and $tic;
    $GVKEY_TO_TIC{$gvkey} = $tic;

}

close(FP);

while(<STDIN>) {

    chomp;
    push @lines, $_;

}

my $skip_idx = 0;

while( @lines ) {

    $skip_idx++;

    my @recs = ( );

    for ( 1 .. 9 ) {

	push @recs, shift @lines;

    }

    my ($date,$key,$mse) = split ' ',$recs[0];

    next if $date < $min_date;

    next if ($skip_idx % 3);

    my $tic = $GVKEY_TO_TIC{$key};

    print("$tic t-1=$date $mse\n");
    for my $idx ( 1 .. $#recs ) {
	print($recs[$idx],"\n");
    }

}


