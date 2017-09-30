#! /usr/bin/env perl

use strict;

main();

sub main {

    $| = 1;
    
    my $start_date = $ARGV[0] || die "Must provide start date";
    my $end_date   = $ARGV[1] || die "Must provide end date";

    my @headers = ();
    my @lines = ();

    LINE:
    while( <STDIN> ) {

	chomp;
	my @fields = split ' ',$_;
	
	if ($fields[0] eq 'gvkey') {
	    @headers = @fields;
	    next LINE;
	}

	my $date = $fields[1];

	if (($date ge $start_date) && ($date le $end_date)) {
	    push @lines, \@fields;
	}

    }
    print(join(' ',@headers),"\n") if scalar(@headers);

    foreach my $line (@lines) {
	
	print(join(' ',@$line),"\n");

    }

}
