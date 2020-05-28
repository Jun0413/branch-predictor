echo "fp_1"
bunzip2 -kc ../traces/fp_1.bz2 | ../src/predictor --gshare:13
echo "fp_2"
bunzip2 -kc ../traces/fp_2.bz2 | ../src/predictor --gshare:13
echo "int_1"
bunzip2 -kc ../traces/int_1.bz2 | ../src/predictor --gshare:13
echo "int_2"
bunzip2 -kc ../traces/int_2.bz2 | ../src/predictor --gshare:13
echo "mm_1"
bunzip2 -kc ../traces/mm_1.bz2 | ../src/predictor --gshare:13
echo "mm_2"
bunzip2 -kc ../traces/mm_2.bz2 | ../src/predictor --gshare:13