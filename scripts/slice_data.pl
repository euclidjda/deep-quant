#! /usr/bin/env perl

# Purpose: Create a file with data point between start_date and end_date, inclusive
# Date format is YYYYMM
#
# Usage: slice_data.pl start_date end_date < file_with_data.dat > sliced_data_file.dat
#
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
	
	if ($fields[1] eq 'gvkey') {
	    @headers = @fields;
	    next LINE;
	}

	my $date = $fields[0];

	if (($date ge $start_date) && ($date le $end_date)) {
	    push @lines, \@fields;
	}

    }
    print(join(' ',@headers),"\n") if scalar(@headers);

    foreach my $line (@lines) {
	
	print(join(' ',@$line),"\n");

    }

}
