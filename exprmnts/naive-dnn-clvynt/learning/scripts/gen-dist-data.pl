#!/usr/local/bin/perl -w

use strict;

$| = 1;

my @lines = ();

my $min_date = $ARGV[0] || '190001';

while(<STDIN>) {

    chomp;
    push @lines, $_;

}

print "date key target output\n";

while( @lines ) {

    my @recs = ( );

    for ( 1 .. 9 ) {

	push @recs, shift @lines;

    }

    my ($date,$key) = split ' ',$recs[0];

    next if $date < $min_date;

    my @outputs = split(' ',$recs[6]);
    my @targets = split(' ',$recs[7]);

    my $output = $outputs[4];
    my $target = $targets[4];

    if (defined($target) && defined($output)) {

	print $date," ",$key," ",$target," ",$output,"\n";

    } else {

	print STDERR $key," ",$date,"\n";
	print STDERR $recs[6],"\n";
	print STDERR $recs[7],"\n";
	print STDERR "----\n";
    }

}


