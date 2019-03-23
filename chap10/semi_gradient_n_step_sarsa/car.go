package main

import (
	"math"
	"math/rand"
	"sarsa/tile"

	"github.com/cpmech/gosl/plt"
	"github.com/cpmech/gosl/utl"
	"github.com/schollz/progressbar"
)

const (
	positionMin = -1.2
	positionMax = 0.5
	velocityMin = -0.07
	velocityMax = 0.07

	numTilings    = 8
	positionScale = numTilings / (positionMax - positionMin)
	velocityScale = numTilings / (velocityMax - velocityMin)

	size = 2048
)

var w = make([]float64, size, size)

func tileCoding(position, velocity float64, action int) []int {
	a := make([]int, 0)
	f := make([]float64, 2, 2)
	f[0] = position * positionScale
	f[1] = velocity * velocityScale
	i := make([]int, 1, 1)
	i[0] = action
	tile.Tiles(&a, int(numTilings), size, f, i)
	return a
}

func step(position, velocity float64, action int) (float64, float64, float64) {
	newVelocity := velocity + 0.001*float64(action) - 0.0025*math.Cos(3*position)
	newVelocity = math.Min(math.Max(newVelocity, velocityMin), velocityMax)
	newPosition := position + newVelocity
	newPosition = math.Min(math.Max(newPosition, positionMin), positionMax)
	reward := -1.0
	if newPosition == positionMax {
		reward = 0.0
	}
	if newPosition == positionMin {
		newVelocity = 0.0
	}
	return newPosition, newVelocity, reward
}

func reset() {
	w = make([]float64, size, size)
}

func getValue(position, velocity float64, action int) float64 {
	if position >= positionMax {
		return 0.0
	}
	t := tileCoding(position, velocity, action)
	sum := 0.0
	for _, e := range t {
		sum += w[e]
	}
	return sum
}

func argMax(f []float64) int {
	i := 0
	m := f[0]
	for j, e := range f {
		if e > m {
			i = j
		}
	}
	return i
}

func getAction(position, velocity float64) int {
	e := 0.1
	if rand.Float64() <= e {
		return int(rand.Float64()*3.0) - 1 // -1, 0 , 1
	}
	v1 := getValue(position, velocity, -1)
	v2 := getValue(position, velocity, 0)
	v3 := getValue(position, velocity, 1)
	f := make([]float64, 0, 3)
	f = append(f, v1, v2, v3)
	return argMax(f) - 1
}

func episode(alpha float64, steps int) float64 {
	n := float64(steps)
	positions := make([]float64, 0)
	velocities := make([]float64, 0)
	actions := make([]int, 0)
	rewards := make([]float64, 0)
	term := math.Inf(1)
	gamma := 1.0
	position := rand.Float64()*0.2 - 0.6
	velocity := 0.0
	action := getAction(position, velocity)
	positions = append(positions, position)
	velocities = append(velocities, velocity)
	actions = append(actions, action)
	rewards = append(rewards, 0)
	t := 0.0
	var (
		newPosition, newVelocity, reward float64
		newAction                        int
	)
	for {
		if t < term {
			newPosition, newVelocity, reward = step(position, velocity, action)
			rewards = append(rewards, reward)
			positions = append(positions, position)
			velocities = append(velocities, velocity)
			if newPosition >= positionMax {
				term = t + 1
			} else {
				newAction = getAction(newPosition, newVelocity)
				actions = append(actions, action)
			}
		}
		tau := int(t - n + 1)
		if tau >= 0 {
			k := tau + steps
			start := tau + 1
			end := int(math.Min(term, float64(k)))
			g := 0.0
			for i := 0; i < int(end-start+1); i++ {
				f := i + start
				g += math.Pow(gamma, float64(f-tau-1)) * rewards[int(f)]
			}
			if float64(k) < term {
				g += math.Pow(gamma, float64(n)) * getValue(positions[k], velocities[k], actions[k])
			}
			tile := tileCoding(positions[tau], velocities[tau], actions[tau])
			for _, e := range tile {
				w[e] += alpha * (g - getValue(positions[tau], velocities[tau], actions[tau]))
			}
		}
		t++
		position = newPosition
		velocity = newVelocity
		action = newAction
		if tau == int(term-1) {
			break
		}
	}
	return t
}

func run() {
	count := 100
	const episodes = 500
	alpha1 := 0.5 / numTilings
	alpha2 := 0.3 / numTilings
	n1 := 1
	n2 := 8
	steps1 := make([]float64, episodes, episodes)
	steps2 := make([]float64, episodes, episodes)
	bar := progressbar.New(count)
	for i := 0; i < count; i++ {
		reset()
		for j := 0; j < episodes; j++ {
			steps1[j] += episode(alpha1, n1)
		}
		reset()
		for j := 0; j < episodes; j++ {
			steps2[j] += episode(alpha2, n2)
		}
		bar.Add(1)
	}

	for i := 0; i < episodes; i++ {
		steps1[i] = steps1[i] / float64(count)
		steps2[i] = steps2[i] / float64(count)
	}

	data := make([][]float64, 0, 2)
	data = append(data, steps1, steps2)
	labels := make([]string, 0, 2)
	labels = append(labels, "alpha=0.5/8,n=1", "alpha=0.3/8,n=8")

	colors := []string{"blue", "orange"}
	image(episodes, data, labels, colors)

}

func image(episodes int, data [][]float64, labels []string, colors []string) {
	// data
	x := utl.LinSpace(0.0, float64(episodes), episodes)
	// clear figure
	plt.Reset(false, nil)
	for i, e := range data {
		plt.Plot(x, e, &plt.A{C: colors[i], L: labels[i]})
	}
	l := make([]*plt.A, 0)
	for i, e := range labels {
		l = append(l, &plt.A{C: colors[i], L: e})
	}
	plt.LegendX(l, nil)
	limits := make([]float64, 0, 4)
	limits = append(limits, 0.0, 500.0, 100.0, 1000.0)
	plt.AxisLims(limits)
	plt.SetLabels("episodes", "steps per episode", nil)
	plt.Save("./", "steps")
}

func main() {
	run()
}
