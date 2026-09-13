package main

// a base tokenizer class
type Tokenizer struct {
	Merges        map[Pair]int
	Pattern       string
	SpecialTokens map[string]int
	Vocab         map[int][]byte
}

// initializer for tokenizer
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		Merges:        make(map[Pair]int),
		Pattern:       "",
		SpecialTokens: make(map[string]int),
	}
	t.Vocab = t.BuildVocab()
	return t
}

// buildvocab will build the vocabulary map derived from base bytes, merges and special tokens.
func (t *Tokenizer) BuildVocab() map[int][]byte {
	vocab := make(map[int][]byte)

	//default base vocabulary for all 256 byte values
	for idx := 0; idx < 256; idx++ {
		vocab[idx] = []byte{byte(idx)}
	}

	//append merged tokens
	for pair, idx := range t.Merges {
		merged := make([]byte, 0, len(vocab[pair.A])+len(vocab[pair.B]))
		merged = append(merged, vocab[pair.A]...)
		merged = append(merged, vocab[pair.B]...)
		vocab[idx] = merged
	}

	// append special tokens encoded as UTF-8 bytes
	for special, idx := range t.SpecialTokens {
		vocab[idx] = []byte(special)
	}

	return vocab
}
