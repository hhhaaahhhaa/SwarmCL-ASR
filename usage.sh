#!bin/bash
python one.py -s wav2vec2 -t cv-aus -n exp0/aus
python one.py -s wav2vec2 -t cv-eng -n exp0/eng
python one.py -s wav2vec2 -t cv-ind -n exp0/ind
python one.py -s wav2vec2 -t cv-ire -n exp0/ire
python one.py -s wav2vec2 -t cv-sco -n exp0/sco

python run_benchmark.py -o single/aus -s wav2vec2 -c results/exp0/aus/ckpt/last.ckpt --loader lightning
python run_benchmark.py -o single/eng -s wav2vec2 -c results/exp0/eng/ckpt/last.ckpt --loader lightning
python run_benchmark.py -o single/ind -s wav2vec2 -c results/exp0/ind/ckpt/last.ckpt --loader lightning
python run_benchmark.py -o single/ire -s wav2vec2 -c results/exp0/ire/ckpt/last.ckpt --loader lightning
python run_benchmark.py -o single/sco -s wav2vec2 -c results/exp0/sco/ckpt/last.ckpt --loader lightning

python run_strategy.py -s uniform-soup -t none -n exp0/merging/uniform-soup
python run_strategy.py -s greedy-soup -t cv-val100 -n exp0/merging/greedy-soup

python run_benchmark.py -o merging/avg -s wav2vec2 -c results/exp0/merging/uniform-soup/ckpt/merged.ckpt
python run_benchmark.py -o merging/greedy-soup -s wav2vec2 -c results/exp0/merging/greedy-soup/ckpt/merged.ckpt
