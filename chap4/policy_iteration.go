/*
MIT License

Copyright (c) 2018 Frank Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"time"
)

/*
probability of requested and returned cars is Poisson random variables
P(n) = lambda^n * e^{-lambda} / n!
lambda is 3 and 4 for rental requests at first and second location
lambda is 3 and 2 for returns

there can be no more than 20 cars at each location

rent a car out is credited $10
move a car costs $2
at most five cars can be moved from one location to the other one night

gamma = 0.9

cars are availible the day after they are returned

state is the number of cars at each location at the end of the day.
actions are the net numbers of cars moved between the two locations.
*/

var POISSONCACHE [2][2][21]float64

func initPolicy() *[21][21]int {
	var policy [21][21]int
	for i := 0; i < 21; i++ {
		for j := 0; j < 21; j++ {
			policy[i][j] = 0
		}
	}
	return &policy
}

// poisson distribution
// P(n) = lambda^n * e^{-lambda} / n!
func poisson(lambda, n int) float64 {
	r := 1
	for i := 0; i < n; i++ {
		r *= (i + 1)
	}
	return math.Pow(float64(lambda), float64(n)) *
		math.Exp(float64(-lambda)) /
		float64(r)
}

func initPoissonCache() {
	for i := 0; i < 21; i++ {
		POISSONCACHE[0][0][i] = poisson(3, i) // first location request
		POISSONCACHE[1][0][i] = poisson(4, i) // second location request
		POISSONCACHE[0][1][i] = poisson(3, i) // first location return
		POISSONCACHE[1][1][i] = poisson(2, i) // second location return
	}
}

func initValue() *[21][21]float64 {
	var value [21][21]float64
	for i := 0; i < 21; i++ {
		for j := 0; j < 21; j++ {
			value[i][j] = rand.Float64()
		}
	}
	return &value
}

// getAllActions get all choices under certain circumstance
// minimum is negtive of the number of cars in the second location
// maximum is the number of cars in first location
func getAllActions(state *[2]int) *[]int {
	a := state[0]
	if a >= 5 {
		a = 5
	}
	b := state[1]
	if b >= 5 {
		b = 5
	}
	actions := make([]int, 0, a+b+1)
	for i := -b; i <= a; i++ {
		actions = append(actions, i)
	}
	return &actions
}

func getPolicyActions(state *[2]int, policy *[21][21]int) int {
	return policy[state[0]][state[1]]
}

func getPoissonProb(l, r, n int) float64 {
	return POISSONCACHE[l][r][n]
}

// truncate
// there are no more than 20 cars in each location, and number cannot be negative
func truncate(i int) int {
	if i < 0 {
		return 0
	}
	if i > 20 {
		return 20
	}
	return i
}

type condition struct {
	state  *[2]int
	reward float64
	prob   float64
}

func step(state *[2]int, action int) []*condition {
	s0 := truncate(state[0] - action)
	s1 := truncate(state[1] + action)
	r0 := int(-2 * math.Abs(float64(action)))
	m := make([]*condition, 0, 21*21*21*21)
	for i1 := 0; i1 <= 20; i1++ { // return of first location
		for j1 := 0; j1 <= truncate((s0 + i1)); j1++ { // request of first location, no more than stock
			for i2 := 0; i2 <= 20; i2++ { // return of second location
				for j2 := 0; j2 <= truncate((s1 + i2)); j2++ { // request of second location, no more than stock
					s := &[2]int{truncate(s0 + i1 - j1), truncate(s1 + i2 - j2)}
					r1 := int(math.Min(float64(j1), float64(s0+i1)))
					r2 := int(math.Min(float64(j2), float64(s1+i2)))
					r := float64(10*(r1+r2) + r0)
					p := getPoissonProb(0, 0, j1) * getPoissonProb(0, 1, i1) * getPoissonProb(1, 0, j2) * getPoissonProb(1, 1, i2)
					m = append(m, &condition{s, r, p})
				}
			}
		}
	}
	return m
}

func expected(conds []*condition, value *[21][21]float64, state *[2]int) float64 {
	result := 0.0
	gamma := 0.9
	for _, c := range conds {
		result += c.prob * (c.reward + gamma*value[c.state[0]][c.state[1]])
	}
	return result
}

func policyEvaluation(value *[21][21]float64, policy *[21][21]int) {
	theta := 0.0001
	c := 0
	start := time.Now()
	for {
		c += 1
		m := 0.0
		for i := 0; i < 21; i++ {
			for j := 0; j < 21; j++ {
				v := value[i][j]
				s := &[2]int{i, j}
				actions := getPolicyActions(s, policy)
				conds := step(s, actions)
				value[i][j] = expected(conds, value, s)
				m = math.Max(m, math.Abs(v-value[i][j]))
			}
		}
		current := time.Now()
		cost := current.Sub(start)
		start = current
		fmt.Printf("%d max change during policy evaluation %f cost: %f\n", c, m, cost.Seconds()) // 5.62s
		if m < theta {
			break
		}
	}
}

func policyImprovement(value *[21][21]float64, policy *[21][21]int) bool {
	changed := false
	start := time.Now()
	c := 0
	for i := 0; i < 21; i++ {
		for j := 0; j < 21; j++ {
			state := &[2]int{i, j}
			m := 0.0
			ra := policy[i][j]
			ma := 0
			actions := getAllActions(state)
			for _, a := range *actions {
				conds := step(state, a)
				v := expected(conds, value, state)
				if v > m {
					m = v
					ma = a
				}
			}
			if ra != ma {
				c++
				policy[i][j] = ma
				changed = true
			}
			current := time.Now()
			cost := current.Sub(start)
			start = current
			fmt.Printf("%d changes in policy_improvement cost: %f\n", c, cost.Seconds())
		}

	}
	return changed
}

func output(policy *[21][21]int) {
	f, _ := os.Create("./policy.txt")
	defer f.Close()
	for i := 0; i < 21; i++ {
		for j := 0; j < 21; j++ {
			s := fmt.Sprintf("%d, %d : %d\n", i, j, policy[i][j])
			f.WriteString(s)
		}
	}
}

func getDataIdx(w, h, times int) (int, int) {
	i := 0
	for w >= (i+1)*times {
		i++
	}
	j := 0
	for h >= (j+1)*times {
		j++
	}
	return i, j
}

func save(data *[21][21]int, times int) {
	margin := 15
	length := 21 * times
	width := length + 2*margin
	height := length + 2*margin
	f, _ := os.Create("./policy.png")
	defer f.Close()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for i := 0; i < margin; i++ {
		for j := 0; j < height; j++ {
			img.Set(i, j, color.White)
			img.Set(i+length+margin, j, color.White)
		}
	}
	for i := 0; i < length; i++ {
		for j := 0; j < margin; j++ {
			img.Set(i+margin, j, color.White)
			img.Set(i+margin, j+length+margin, color.White)
		}
	}
	for x := 0; x < length; x++ {
		for y := 0; y < length; y++ {
			a, b := getDataIdx(x, y, times)
			d := data[a][b] // {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
			if d > 0 {
				rate := float64(d+5) / float64(11.0)
				c := uint8(float64(256) * rate)
				img.Set(x+margin, y+margin, color.RGBA{255, c, 255, 255})
			} else if d == 0 {
				img.Set(x+margin, y+margin, color.White)
			} else {
				rate := float64(d+5) / float64(11.0)
				c := uint8(float64(256) * rate)
				img.Set(x+margin, y+margin, color.RGBA{c, c, c, 255})
			}

		}
	}
	png.Encode(f, img)
}

func run() *[21][21]int {
	b := true
	initPoissonCache()
	policy := initPolicy()
	value := initValue()
	c := 0
	for b {
		c++
		policyEvaluation(value, policy)
		fmt.Println("policy evaluatoin")
		b = policyImprovement(value, policy)
		fmt.Println("policy improvement")
		fmt.Println(c, "--------------------------------------------------------")
	}
	return policy
}

func main() {
	policy := run()
	fmt.Println("over")
	output(policy)
	save(policy, 50)
}
