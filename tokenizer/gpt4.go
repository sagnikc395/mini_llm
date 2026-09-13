package main

import (
	"bytes"
	"math"
)

func BPE(mergeableRanks map[string]int, token []byte, maxRank *int) [][]byte {
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
		if minIdx == -1 || (maxRank != nil && minRank >= *maxRank) {
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
