package main

// base tokenizer class and some common helper functions.
// base tokenizer class also contains the (common) save/load functionality

// pair represents a consecutive pair of integers
type Pair struct {
	A, B int
}

// GetStats takes a slice of integers and a optional existing map
// if counts are nil, then a new map will be initialized
func GetStats(ids []int, counts map[Pair]int) map[Pair]int {

	//check if a nil or not
	if counts == nil {
		counts = make(map[Pair]int)
	}

	for i := 0; i < len(ids)-1; i++ {
		pair := Pair{ids[i], ids[i+1]}
		counts[pair] += 1
	}

	return counts

}
