#!/usr/local/bin/perl -w

use strict;

$| = 1;

my $DATE_IDX   = 0;
my $KEY_IDX    = 1;
my $TIC_IDX    = 2;
my $CSHO_IDX   = 8;
my $TARGET_IDX = 13;
my $OUTPUT_IDX = 14;

my @lines = ();

my $min_date = $ARGV[0] || '190001';
my $max_date = $ARGV[1] || '999999';

while(<STDIN>) {

    chomp;
    push @lines, $_;

}

my %results = ();
my $missing_count = 0;

while( @lines ) {

    my $line = shift @lines;

    my $date   = $lines[$DATE_IDX];
    my $key    = $lines[$KEY_IDX];
    my $tic    = $lines[$TIC_IDX];
    my $csho   = $lines[$CSHO_IDX];
    my $target = $lines[$TARGET_IDX];
    my $output = $lines[$OUTPUT_IDX];

    die "No value for date" unless $date =~ /^\d+$/;
    die "No value for key" unless $key =~ /^\d+$/;

    next unless ($csho ne 'NULL' && $csho =~ /^\d+$/);

    if ($target eq 'NULL' || $output eq 'NULL') {
	$missing_count++ if $tic ne '_CASH';
	next;
    }

}




