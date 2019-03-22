package test

import (
	"testing"
	"tile"
)

func TestTile(t *testing.T) {
	tiles := make([]int, 0)
	f := make([]float64, 0, 2)
	f = append(f, 0.3)
	f = append(f, 0.9)
	i := make([]int, 0, 1)
	i = append(i, 2)
	tile.Tiles(&tiles, 8, 2048, f, i)
	t.Log(tiles)
}
