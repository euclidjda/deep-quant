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

	for my $optimizer ('AdadeltaOptimizer','AdagradOptimizer','GradientDescentOptimizer') {

	    for my $input_dropout ('True','False') {

		for my $hidden_dropout ('True','False') {

		    for my $target_lambda (0.0,0.25,0.5,0.75,1.0) {

			my @rnn_lambdas = (($model_type eq 'DeepRnnModel') && ($target_lambda < 1.0)) ? (0.5,0.7,0.9,1.0) : (1.0);

			for my $rnn_lambda (@rnn_lambdas) {
			
			    for my $num_layers (1,2,4) {

				my @hidden_list = (64,128,512,1024);
				
				for my $num_hidden (@hidden_list) {
				    
				    my @keep_prob_list = (1.0);
				    
				    @keep_prob_list = (0.5,0.75)
					if (($input_dropout eq 'True') || ($hidden_dropout eq 'True'));
				    
				    for my $keep_prob (@keep_prob_list) {
					
					for my $init_scale (0.01,0.1) {
					    
					    for my $max_norm (0,1,5,10,50,100) {
						
						my $name = sprintf("%s-%s-%s-l%d-h%04d-t%03d-r%02d-k%02d-i%02d-m%03d-%s%s",
								   substr(lc($model_type),4,3),
								   substr(lc($optimizer),0,4),
								   substr(lc($scaler),0,4),
								   $num_layers,
								   $num_hidden,
								   int($target_lambda*100),
								   int($rnn_lambda*10),
								   int($keep_prob*10),
								   int($init_scale*100),
								   $max_norm,
								   lc(substr($input_dropout,0,1)),
								   lc(substr($hidden_dropout,0,1)));
						
						push(@names,$name);

my $CONFIG_STR =<<"CONFIG_STR";
--default_gpu		/gpu:0
--key_field             gvkey
--active_field          active
--target_field		oiadpq_ttm
--feature_fields        saleq_ttm-ltq_mrq
--aux_input_fields      mom1m-mom9m
--scale_field           mrkcap
--datafile		source-ml-data-100M-train.dat
--data_dir		datasets
--model_dir		$CHKPTS_DIR/chkpts-$name
--nn_type		$model_type
--optimizer	        AdadeltaOptimizer
--data_scaler           $scaler
--passes		0.2
--stride                12
--num_unrollings        5
--batch_size		256
--validation_size	0.30
--seed                  521
--max_epoch		10000
--early_stop		25
--keep_prob		$keep_prob
--learning_rate         0.6
--lr_decay		0.95
--init_scale		$init_scale
--max_grad_norm		$max_norm
--num_layers            $num_layers
--num_hidden		$num_hidden
--skip_connections      True
--input_dropout         $input_dropout
--hidden_dropout        $hidden_dropout
--rnn_lambda            $rnn_lambda
--target_lambda         $target_lambda
--cache_id              1024
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
    }
    
}

@names = shuffle(@names);

foreach my $name (@names) {

    printf("deep_quant.py --config=$CONFIG_DIR/$name.conf > $OUTPUT_DIR/stdout-$name.txt 2> $OUTPUT_DIR/stderr-$name.txt ; rm -rf $CHKPTS_DIR/chkpts-$name ; \n");

}

