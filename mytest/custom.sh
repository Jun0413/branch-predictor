echo "fp_1"
bunzip2 -kc ../traces/fp_1.bz2 | ../src/predictor --custom
echo "fp_2"
bunzip2 -kc ../traces/fp_2.bz2 | ../src/predictor --custom
echo "int_1"
bunzip2 -kc ../traces/int_1.bz2 | ../src/predictor --custom
echo "int_2"
bunzip2 -kc ../traces/int_2.bz2 | ../src/predictor --custom
echo "mm_1"
bunzip2 -kc ../traces/mm_1.bz2 | ../src/predictor --custom
echo "mm_2"
bunzip2 -kc ../traces/mm_2.bz2 | ../src/predictor --custom