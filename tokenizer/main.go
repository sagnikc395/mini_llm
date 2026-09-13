package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

func main() {
	// read the file for tokenization
	path := filepath.Join(os.TempDir(), "dat")
	dat, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}

	// create a new directory to store the files
	err = os.Mkdir("models", 0755)
	if err != nil {
		log.Fatal(err)
	}

	now := time.Now()

	// training loop
	// TODO!
	//
	then := time.Now()

	fmt.Sprintf("Training took {%d - %d:.2f} seconds", then, now)
}
