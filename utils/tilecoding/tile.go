/*
Package tile    tile coding
this code is translated from c++ in http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/steph/CTiles050125.zip
*/
package tile

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

const (
	// MaxNumVars Maximum number of variables in a grid-tiling
	MaxNumVars = 20
	// MaxNumCoords Maximum number of hashing coordinates
	MaxNumCoords = 100
	MaxLongInt   = 2147483647
)

var firstCall = true
var rndseq = make([]int, 2048, 2048)

// Tiles tiles
// theTiles     provided array contains returned tiles (tile indices)
// numTilings   number of tile indices to be returned in tiles
// memorySize   total number of possible tiles
// floats       array of floating point variables
// ints         array of integer variables
func Tiles(theTiles *[]int, numTilings int, memorySize int, floats []float64, ints []int) {
	var (
		i, j   int
		qstate [MaxNumVars]int
		base   [MaxNumVars]int
	)
	coordinates := make([]int, MaxNumVars*2+1, MaxNumVars*2+1) // one interval number per relevant dimension
	numCoordinates := len(floats) + len(ints) + 1
	for i = 0; i < len(ints); i++ {
		coordinates[len(floats)+i+1] = ints[i]
	}

	// quantize state to integers (henceforth, tile widths == num_tilings)
	for i = 0; i < len(floats); i++ {
		qstate[i] = int(math.Floor(floats[i] * float64(numTilings)))
		base[i] = 0
	}

	// compute the tile numbers
	for j = 0; j < numTilings; j++ {
		// loop over each relevant dimension
		for i = 0; i < len(floats); i++ {
			// find coordinates of activated tile in tiling space
			if qstate[i] >= base[i] {
				coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % numTilings)
			} else {
				coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % numTilings) - numTilings
			}
			// compute displacement of next tiling in quantized space
			base[i] += 1 + (2 * i)
		}
		// add additional indices for tiling and hashing_set so they hash differently
		coordinates[i] = j
		*theTiles = append(*theTiles, hashUNH(coordinates, numCoordinates, memorySize, 449))
	}
}

func hashUNH(ints []int, numInts, m, increment int) int {
	var i, k, index int
	sum := 0

	// if first call to hashing, initialize table of random numbers
	if firstCall {
		for k = 0; k < 2048; k++ {
			rndseq[k] = 0
			for i = 0; i < strconv.IntSize/8; i++ {
				rndseq[k] = (rndseq[k] << 8) | (rand.Int() & 0xff)
				// rndseq[k] = (rndseq[k] << 8) | (3 & 0xff)
			}
		}
		firstCall = false
	}

	for i = 0; i < numInts; i++ {
		// add random table offset for this dimension and wrap around
		index = ints[i]
		index += (increment * i)
		// index %= 2048
		index = index & 2047
		for index < 0 {
			index += 2048
		}

		// add selected random number to sum
		sum += rndseq[index]
	}
	index = sum % m
	for index < 0 {
		index += m
	}

	return index
}

// Tiles2 tiles using collision table
// theTiles    provided array contains returned tiles (tile indices)
// numTilinfs  number of tile indices to be returned in tiles
// ctable      total number of possible tiles
// floats      array of floating point variables
// ints        array of integer variables
func Tiles2(theTiles *[]int, numTilings int, ctable *CollisionTable, floats []float64, ints []int) {
	var (
		i, j   int
		qstate [MaxNumVars]int
		base   [MaxNumVars]int
	)
	coordinates := make([]int, MaxNumVars*2+1, MaxNumVars*2+1) // one interval number per relevant dimension
	numCoordinates := len(floats) + len(ints) + 1

	for i = 0; i < len(ints); i++ {
		coordinates[len(floats)+i+1] = ints[i]
	}

	// quantize state to integers (henceforth, tile widths == num_tilings)
	for i = 0; i < len(floats); i++ {
		qstate[i] = int(math.Floor(floats[i] * float64(numTilings)))
		base[i] = 0
	}

	// compute the tile numbers
	for j = 0; j < numTilings; j++ {
		// loop over each relevant dimension
		for i = 0; i < len(floats); i++ {
			if qstate[i] >= base[i] {
				coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % numTilings)
			} else {
				coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % numTilings) - numTilings
			}
			// compute displacement of next tiling in quantized space
			base[i] += 1 + (2 * i)
		}
		// add additional indices for tiling and hashing_set so they hash differently
		coordinates[i] = j
		*theTiles = append(*theTiles, hash(coordinates, numCoordinates, ctable))
	}
}

// hash  Takes an array of integers and returns the corresponding tile after hashing
func hash(ints []int, numInts int, ct *CollisionTable) int {
	var j, ccheck int
	ct.calls++
	j = hashUNH(ints, numInts, ct.m, 449)
	ccheck = hashUNH(ints, numInts, MaxLongInt, 457)
	if ccheck == ct.data[j] {
		ct.clearhits++
	} else if ct.data[j] == -1 {
		ct.clearhits++
		ct.data[j] = ccheck
	} else if ct.safe == 0 {
		ct.collisions++
	} else {
		h2 := 1 + 2*hashUNH(ints, numInts, (MaxLongInt)/4, 449)
		for i := 1; true; i++ {
			ct.collisions++
			j = (j + h2) % ct.m
			if i > ct.m {
				log.Fatalf("\nTiles: Collision table out of Memory")
			}
			if ccheck == ct.data[j] {
				break
			}
			if ct.data[j] == -1 {
				ct.data[j] = ccheck
				break
			}
		}
	}
	return j
}

type CollisionTable struct {
	m          int
	data       []int
	safe       int
	calls      int
	clearhits  int
	collisions int
}

func (t *CollisionTable) reset() {
	for i := 0; i < t.m; i++ {
		t.data[i] = -1
	}
	t.calls = 0
	t.clearhits = 0
	t.collisions = 0
}

func NewCollisionTable(size, safety int) *CollisionTable {
	tmp := size
	for tmp > 2 {
		if tmp%2 != 0 {
			log.Fatalf("\nSize of collision table must be power of 2 %d", size)
		}
		tmp /= 2
	}
	t := &CollisionTable{
		data: make([]int, size, size),
		m:    size,
		safe: safety,
	}
	t.reset()
	return t
}

func (t *CollisionTable) Usage() int {
	count := 0
	for i := 0; i < t.m; i++ {
		if t.data[i] != -1 {
			count++
		}
	}
	return count
}

func (t *CollisionTable) Print() {
	fmt.Printf("Collision table: Safety : %d Usage : %d Size : %d Calls : %d Collisions : %d\n", t.safe, t.Usage(), t.m, t.calls, t.collisions)
}

func (t *CollisionTable) Save(file string) {
	f, err := os.Create(file)
	if err != nil {
		fmt.Printf("save collision table to %s error: %s", file, err.Error())
		return
	}
	defer f.Close()
	m := make(map[string]interface{})
	m["m"] = t.m
	m["safe"] = t.safe
	m["calls"] = t.calls
	m["clearhits"] = t.clearhits
	m["collisions"] = t.collisions
	m["data"] = t.data[0:t.m]
	b, _ := json.Marshal(m)
	f.Write(b)
}

func (t *CollisionTable) Restore(file string) {
	f, err := os.Open(file)
	if err != nil {
		fmt.Printf("restore collision table from %s error: %s", file, err.Error())
		return
	}
	defer f.Close()
	b, err := ioutil.ReadAll(f)
	if err != nil {
		fmt.Printf("restore collision table from %s error: %s", file, err.Error())
		return
	}
	var m map[string]interface{}
	err = json.Unmarshal(b, &m)
	if err != nil {
		fmt.Printf("restore collision table from %s error: %s", file, err.Error())
		return
	}
	t.m = int(m["m"].(float64))
	t.safe = int(m["safe"].(float64))
	t.calls = int(m["calls"].(float64))
	t.clearhits = int(m["clearhits"].(float64))
	t.collisions = int(m["collisions"].(float64))
	d := m["data"].([]interface{})
	data := make([]int, t.m, t.m)
	for i, e := range d {
		data[i] = int(e.(float64))
	}
	t.data = data
}

// TilesWrap tiles wrap
// theTiles       provided array contains returned tiles (tile indices)
// numTilings     number of tile indices to be returned in tiles
// memorySize     total number of possible tiles
// floats         array of floating point variables
// wrapWidths     array of widths (length and units as in floats)
// ints           array of integer variables
func TilesWrap(theTiles *[]int, numTilings, memorySize int, floats []float64, wrapWidths, ints []int) {
	var (
		i, j                     int
		qstate                   [MaxNumVars]int
		base                     [MaxNumVars]int
		wrapWidthsTimeNumTilings [MaxNumVars]int
	)
	// one interval number per relevant dimension
	coordinates := make([]int, MaxNumVars*2+1, MaxNumVars*2+1)
	numCoordinates := len(floats) + len(ints) + 1

	for i = 0; i < len(ints); i++ {
		coordinates[len(floats)+1+i] = ints[i]
	}

	// quantize state to integers (henceforth, tile widths == num_tilings)
	for i = 0; i < len(floats); i++ {
		qstate[i] = int(math.Floor(floats[i] * float64(numTilings)))
		base[i] = 0
		wrapWidthsTimeNumTilings[i] = wrapWidths[i] * numTilings
	}
	// compute the tile numbers
	for j = 0; j < numTilings; j++ {
		// loop over each relevant dimension
		for i = 0; i < len(floats); i++ {
			// find coordinates of activated tile in tiling space
			if qstate[i] >= base[i] {
				coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % numTilings)
			} else {
				coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % numTilings) - numTilings
			}
			if wrapWidths[i] != 0 {
				coordinates[i] = coordinates[i] % wrapWidthsTimeNumTilings[i]
			}
			if coordinates[i] < 0 {
				for coordinates[i] < 0 {
					coordinates[i] += wrapWidthsTimeNumTilings[i]
				}
			}
			// compute displacement of next tiling in quantized space
			base[i] += 1 + (2 * i)
		}
		// add additional indices for tiling and hashing_set so they hash differently
		coordinates[i] = j
		*theTiles = append(*theTiles, hashUNH(coordinates, numCoordinates, memorySize, 449))
	}
}

// TilesWrap2 tiles wrap
// theTiles       provided array contains returned tiles (tile indices)
// numTilings     number of tile indices to be returned in tiles
// ctable         total number of possible tiles
// floats         array of floating point variables
// wrapWidths     array of widths (length and units as in floats)
// ints           array of integer variables
func TilesWrap2(theTiles *[]int, numTilings int, ctable *CollisionTable, floats []float64, wrapWidths, ints []int) {
	var (
		i, j                     int
		qstate                   [MaxNumVars]int
		base                     [MaxNumVars]int
		wrapWidthsTimeNumTilings [MaxNumVars]int
	)
	// one interval number per relevant dimension
	coordinates := make([]int, MaxNumVars*2+1, MaxNumVars*2+1)
	numCoordinates := len(floats) + len(ints) + 1

	for i = 0; i < len(ints); i++ {
		coordinates[len(floats)+1+i] = ints[i]
	}

	// quantize state to integers (henceforth, tile widths == num_tilings)
	for i = 0; i < len(floats); i++ {
		qstate[i] = int(math.Floor(floats[i] * float64(numTilings)))
		base[i] = 0
		wrapWidthsTimeNumTilings[i] = wrapWidths[i] * numTilings
	}
	// compute the tile numbers
	for j = 0; j < numTilings; j++ {
		// loop over each relevant dimension
		for i = 0; i < len(floats); i++ {
			// find coordinates of activated tile in tiling space
			if qstate[i] >= base[i] {
				coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % numTilings)
			} else {
				coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % numTilings) - numTilings
			}
			if wrapWidths[i] != 0 {
				coordinates[i] = coordinates[i] % wrapWidthsTimeNumTilings[i]
			}
			if coordinates[i] < 0 {
				for coordinates[i] < 0 {
					coordinates[i] += wrapWidthsTimeNumTilings[i]
				}
			}
			// compute displacement of next tiling in quantized space
			base[i] += 1 + (2 * i)
		}
		// add additional indices for tiling and hashing_set so they hash differently
		coordinates[i] = j
		*theTiles = append(*theTiles, hash(coordinates, numCoordinates, ctable))
	}
}
