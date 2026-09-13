package main

import "fmt"

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

func (t *Tokenizer) Train(text string, vocab_size int, verbose bool) {
	// tokenizer can train a vocabulary of size vocab_size from text
	panic("Not Implemented Error")
}

func (t *Tokenizer) Encode(text string) {
	// tokenizer can encode a string into a list of integers
	panic("Not Implemented Error")
}

func (t *Tokenizer) Decode(ids []int) {
	// tokenizer can decode a list of integers into a string
	panic("Not Implemented Error")
}

func (t *Tokenizer) Save(file_prefix string) {
	/**
	saves two files : file_prefix.vocab and file_prefix.model
	 - inspired by sentencepiece model saving
	 - model file is the critical file, intended for Load()
	 - vocab file is just a pretty printed version for human inspection
	*/

	model_file := file_prefix + ".model"

	// open the model file and write the stuff
	// TODO!

	vocab_file := file_prefix + ".vocab"
	var inverted_merges map[int]Pair
	for pair, idx := range t.Merges() {
		inverted_merges[idx] = pair
	}

	// write the vocab: for the human to look at
	// TODO!
}

func (t* Tokenizer) Load(model_file: string) {
	//inverse of Save() , but only for the model file

	// read the model file
	var merges map[Pair]int
	var special_tokens map[string]int
	idx := 256

	// open file and do the operations
	// TODO!


	t.Merges = merges
	t.SpecialTokens = special_tokens
	t.vocab = t.BuildVocab()
}
