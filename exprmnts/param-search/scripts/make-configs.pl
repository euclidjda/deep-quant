#!/usr/bin/perl -w

use strict;
use FileHandle;
use List::Util qw(shuffle);

my $OUTPUT_DIR = "output";
my $CONFIG_DIR = "config";
my $CHKPTS_DIR = "chkpts";

my @names = ();

for my $model_type ('DeepMlpModel','DeepRnnModel') {

    for my $scaler ('StandardScaler','RobustScaler') {

	for my $input_dropout ('True','False') {

	    for my $hidden_dropout ('True','False') {

		my @lambdas = $model_type eq 'DeepRnnModel' ? (0.5,0.7,0.9,1.0) : (1.0);

		for my $rnn_lambda (@lambdas) {

		    for my $num_layers (1,2,4) {

			for my $num_hidden (128,512,1024) {

			    my @keep_prob_list = (1.0);
			    
			    push @keep_prob_list, (0.5,0.75)
				if (($input_dropout eq 'True') || ($hidden_dropout eq 'True'));

			    for my $keep_prob (@keep_prob_list) {

				for my $init_scale (0.01,0.1) {

				    for my $max_norm (0,1,5,10,50,100) {

					my $name = sprintf("%s-%s-l%d-h%04d-r%04d-k%04d-i%04d-m%04d-%s%s",
							   substr(lc($model_type),4,3),
							   substr(lc($scaler),0,4),
							   $num_layers,
							   $num_hidden,
							   int($rnn_lambda*10),
							   int($keep_prob*10),
							   int($init_scale*100),
							   $max_norm,
							   lc(substr($input_dropout,0,1)),
							   lc(substr($hidden_dropout,0,1)));
					
					push(@names,$name);

my $CONFIG_STR =<<"CONFIG_STR";
--default_gpu		/gpu:0
--nn_type		$model_type
--optimizer	        AdagradOptimizer
--key_field		gvkey
--target_field		target
--scale_field           mrkcap
--active_field          active
--feature_fields        saleq_ttm-ltq_mrq
--data_scaler           $scaler
--datafile		source-ml-data-100M-train.dat
--data_dir		datasets
--model_dir		$CHKPTS_DIR/chkpts-$name
--seed                  1010101
--max_epoch		10000
--early_stop		50
--initial_learning_rate 0.6
--max_grad_norm		$max_norm
--init_scale		$init_scale
--keep_prob		$keep_prob
--passes		0.2
--stride                12
--batch_size		256
--num_unrollings        5
--num_hidden		$num_hidden
--num_layers		$num_layers
--validation_size	0.30
--skip_connections      True
--input_dropout         $input_dropout
--hidden_dropout        $hidden_dropout
--rnn_lambda            $rnn_lambda
CONFIG_STR

my $fh = FileHandle->new("> $CONFIG_DIR/$name.conf");
					$fh->autoflush(1);
					
					print $fh $CONFIG_STR;
					close($fh);
					
				    }
				}
			    }
			    
			}
		    }	  
		}
	    }
	}
    }

}

@names = shuffle(@names);

foreach my $name (@names) {

    printf("deep_quant.py --config=$CONFIG_DIR/$name.conf > $OUTPUT_DIR/stdout-$name.txt 2> $OUTPUT_DIR/stderr-$name.txt ; rm -rf $CHKPTS_DIR/chkpts-$name ; \n");

}

