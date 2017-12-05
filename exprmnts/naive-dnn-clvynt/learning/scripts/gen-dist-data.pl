#!/usr/local/bin/perl -w

use strict;

$| = 1;

my @lines = ();

my $min_date = $ARGV[0] || '190001';
my $max_date = $ARGV[1] || '999999';

while(<STDIN>) {

    chomp;
    push @lines, $_;

}


my %targets = ();
my %outputs = ();

while( @lines ) {

    my @recs = ( );

    for ( 1 .. 9 ) {

	push @recs, shift @lines;

    }

    my ($date,$key) = split ' ',$recs[0];

    next if $date < $min_date;
    next if $date > $max_date;

    # april only
    next unless (substr($date,4,2) eq '04');

    my @outputs = split(' ',$recs[6]);
    my @targets = split(' ',$recs[7]);

    my $output = $outputs[4];
    my $target = $targets[4];

    if (defined($target) && defined($output)) {

	$targets{$date} = 0 unless exists($targets{$date});
	$outputs{$date} = 0 unless exists($outputs{$date});

	$targets{$date} += $target;
	$outputs{$date} += $output;

    } else {

	print STDERR $key," ",$date,"\n";
	print STDERR $recs[6],"\n";
	print STDERR $recs[7],"\n";
	print STDERR "----\n";
    }

}

foreach my $key (sort keys %targets) {

    printf("$key %.2f %.2f\n",$outputs{$key},$targets{$key});

}


