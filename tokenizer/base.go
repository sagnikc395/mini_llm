package main

import "unicode"
import "fmt"
import "unicode/utf8"
import "strings"

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

//Merge methods will replace all consecutive occurences of pair with the new
// integer token idx
// ids = [1,2,3,1,2] , pair = (1,2) idx =4 -> [4,3,4]

func Merge(ids []int, pair Pair, idx int) []int {
	var newids []int
	i := 0
	for i < len(ids) {
		// if not at the very last position and the pair matches, then replace it
		if (ids[i] == pair.A) && (i < len(ids)-1) && (ids[i+1] == pair.B) {
			newids = append(newids, idx)
			i += 2
		} else {
			newids = append(newids, ids[i])
			i += 1
		}
	}
	return newids
}

// ReplaceControlCharacters help us not to print the control characters
// which distort the output ( eg: \n or much worse)

func ReplaceControlCharacters(s string) string {
	var chars []rune
	for _, char := range chars {

		if !unicode.In(char, unicode.C) {
			chars = append(chars, char)
		} else {
			// format the control character as \uXXXX escape sequences
			escaped := []rune(fmt.Sprintf("\\u%04x", char))
			chars = append(chars, escaped...)
		}
	}

	return string(chars)
}

// pretty print a token

func RenderToken(t []byte) string {
	var builder strings.Builder
	builder.Grow(len(t))

	for i := 0; i < len(t); {
		r, size := utf8.DecodeRune(t[i:])
		i += size

		//unicode.In witll match category C (control characters)
		if unicode.In(r, unicode.C) {
			fmt.Fprintf(&builder, "\\u%04x", r)
		} else {
			builder.WriteRune(r)
		}
	}
	return builder.String()
}
