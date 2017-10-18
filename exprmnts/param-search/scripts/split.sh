split -da 4 -l$((`wc -l < runall.sh`/4)) runall.sh runall --additional-suffix=".sh"
