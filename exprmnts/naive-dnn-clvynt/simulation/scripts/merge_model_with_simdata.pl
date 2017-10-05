#! /usr/bin/env perl

use strict;

my $DATE_FIELD_IDX      = 0;
my $GVKEY_FIELD_IDX     = 1;
my $INFO_FIELDS_END_IDX  = 11;
my $NUM_PRE_FACTORS      = 6;

main();

sub main {

    $| = 1;

    my $num_predicted_factors = undef;
    
    my $datafile = $ARGV[0] || die "First cmd arg must be sim data file.";
    my $predfile = $ARGV[1] || die "Second cmd arg must be predictions file.";

    my %keys_to_factors = ();

    open(F1,"< $predfile");

    while(<F1>) {

	chomp;
	my @fields = split(' ',$_);

	next if $fields[0] eq 'gvkey'; # skip header

	my $date  = shift(@fields);
	my $gvkey = shift(@fields);

	$num_predicted_factors = scalar(@fields) unless defined($num_predicted_factors);
	
	my $factor_fields = join(' ',@fields);
	$keys_to_factors{"$gvkey$date"} = $factor_fields;

    }

    # print($num_predicted_factors,"\n");

    close(F1);

    open(F2,"< $datafile");

    while(<F2>) {

	chomp;
	my @fields = split(' ',$_);

	my $date  = $fields[$DATE_FIELD_IDX];
	my $gvkey = $fields[$GVKEY_FIELD_IDX];

	my $end_idx = $INFO_FIELDS_END_IDX + $NUM_PRE_FACTORS;

	print join(' ',@fields[0..$end_idx]);

	if (exists($keys_to_factors{"$gvkey$date"})) {

	  print(' ',$keys_to_factors{"$gvkey$date"},"\n");

	} else {

	  print(' ',join(' ', ('NULL') x $num_predicted_factors ),"\n" );
	  
	}

      }

    close(F2);

}
