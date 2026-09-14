package main

import (
	"bytes"
	"fmt"
	"math"
)

func BPE(mergeableRanks map[string]int, token []byte, maxRank int) [][]byte {
	// helper function used in GetGPT4Merges() to reconstruct the merge forest

	parts := make([][]byte, len(token))
	for i, b := range token {
		parts[i] = []byte{b}
	}

	for {
		minIdx := -1
		minRank := math.MaxInt

		//search for the adjacent pair with the lowest merge rank
		for i := 0; i < len(parts)-1; i++ {
			//concatenate the adjacent parts to look up in the map
			pair := bytes.Join(parts[i:i+2], nil)
			if rank, exists := mergeableRanks[string(pair)]; exists {
				if rank < minRank {
					minIdx = i
					minRank = rank
				}
			}
		}

		//stop if there are no valid merges exist of if the lowest rank exceeds maxRank
		if minIdx == -1 || (maxRank != -1 && minRank >= maxRank) {
			break
		}

		//merge the parts[minIdx] and parts[minIdx+1]
		merged := append(parts[minIdx], parts[minIdx+1]...)
		newParts := make([][]byte, 0, len(parts)-1)
		newParts = append(newParts, parts[:minIdx]...)
		newParts = append(newParts, merged)
		newParts = append(newParts, parts[minIdx+2:]...)
		parts = newParts
	}
	return parts
}

func RecoverMerges(mergeableRanks map[string]int) map[[2]int]int {
	merges := make(map[[2]int]int)

	for token, rank := range mergeableRanks {
		if len(token) == 1 {
			// skip raw bytes
			continue
		}

		// BPE returns pair as [][]byte
		pair := BPE(mergeableRanks, []byte(token), rank)
		if len(pair) < 2 {
			continue
		}

		// Convert []byte parts to string for lookup in mergeableRanks
		ix0 := mergeableRanks[string(pair[0])]
		ix1 := mergeableRanks[string(pair[1])]

		// Map (ix0, ix1) pair tuple to rank
		merges[[2]int{ix0, ix1}] = rank
	}

	return merges
}
