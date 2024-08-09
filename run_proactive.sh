export HF_HOME=/tmp/tf_cache_zhuohan
CUDA_VISIBLE_DEVICES=$1 python test_proactive.py $@
# python test_proactive.py $@