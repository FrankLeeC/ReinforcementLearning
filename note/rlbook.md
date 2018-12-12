# 记录

以下笔记是读 reinforcement learning: an introduction 过程中记录的。

[TOC]



## chapter2.6 Optimistic Initial Values

### Page 34

> Initial action values can also be used as a simple way to encourage exploration. Suppose
> that instead of setting the initial action values to zero, as we did in the 10-armed testbed,
> we set them all to +5. Recall that the $q_*(a)$ in this problem are selected from a normal
> distribution with mean 0 and variance 1. An initial estimate of +5 is thus wildly optimistic.
> But this optimism encourages action-value methods to explore. Whichever actions are
> initially selected, the reward is less than the starting estimates; the learner switches to
> other actions, being “disappointed” with the rewards it is receiving. The result is that all
> actions are tried several times before the value estimates converge. The system does a
> fair amount of exploration even if greedy actions are selected all the time.

当初始值较大时，比如 +5，由于 

$$
\mathcal{R} \sim \mathcal{N} (q(a), 1) \\
q(a) \sim \mathcal{N}(0, 1)
$$

所以当仅利用时，执行action之后的reward极大概率小于 +5，所以更新后的估计值会变小，当下次仅利用时，会挑选一个估计值最大的，此时会选择其余的arm，所以达到了探索的效果。

这种技巧只能应对stationary problem，无法应对nonstationary problem。



### Page 37

exercise2.7

> Exercise 2.7: Unbiased Constant-Step-Size Trick In most of this chapter we have used
> sample averages to estimate action values because sample averages do not produce the
> initial bias that constant step sizes do (see the analysis leading to (2.6)). However, sample
> averages are not a completely satisfactory solution because they may perform poorly
> on nonstationary problems. Is it possible to avoid the bias of constant step sizes while
> retaining their advantages on nonstationary problems? One way is to use a step size of
> $\beta_n \doteq \alpha/\overline o_n$  (2.8)
> to process the nth reward for a particular action, where $\alpha$ > 0 is a conventional constant
> step size, and $\overline o_n$ is a trace of one that starts at 0:
>  $\overline o_n \doteq \overline o_{n-1}+\alpha(1-\overline o_{n-1}) \quad for\quad n \ge 0, with \quad \overline o_0 = 0$    (2.9)
> Carry out an analysis like that in (2.6) to show that $Q_n$ is an exponential recency-weighted
> average without initial bias.

$$
\begin{align} Q_{n+1} & = Q_n + \beta_n(R_n - Q_n) \\
Q_{n+1} & = (1-\beta_n)Q_n+\beta_nR_n \\
Q_{n+1} & = \frac{\alpha}{\overline o_n}R_n + (1-\frac{\alpha}{\overline o_n})Q_n  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n + \frac{\overline o_{n-1} + \alpha(1-\overline o_{n-1}) - \alpha}{\overline o_n}Q_n)  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n + (1-\alpha)\overline o_{n-1} Q_n)  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n + (1-\alpha)\overline o_{n-1}(\frac{\alpha}{\overline o_{n-1}}R_{n-1}+(1-\alpha)\frac{\overline o_{n-2}}{\overline o_{n-1}}Q_{n-1}))  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n+(1-\alpha)\overline o_{n-1}(\frac{1}{\overline o_{n-1}}(\alpha R_{n-1}+(1-\alpha)\overline o_{n-2}Q_{n-1})))  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n+(1-\alpha)\overline o_{n-1}(\frac{1}{\overline o_{n-1}}(\alpha R_{n-1}+(1-\alpha)\overline o_{n-2}(\frac{1}{\overline o_{n-2}}(\alpha R_{n-2}+(1-\alpha)\overline o_{n-3}Q_{n-2})))))  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n+(1-\alpha)\alpha R_{n-1}+(1-\alpha)^2(\alpha R_{n-2} + (1-\alpha)\overline o_{n-3}Q_{n-2}))  \\
Q_{n+1} & = \frac{1}{\overline o_n}(\alpha R_n + (1-\alpha)\alpha R_{n-1}+(1-\alpha)^2\alpha R_{n-2}+(1-\alpha)^3\overline o_{n-3}Q_{n-2})  \\
Q_{n+1} & = \frac{\alpha}{\overline o_n}(R_n + (1-\alpha)R_{n-1}+(1-\alpha)^2R_{n-2}+ ...... + (1-\alpha)^{n-1}\overline R_1 + (1-\alpha)^n\overline o_0Q_1)  \\
\\
因为  \\ 
&\overline o_0 = 0  \\
所以  \\ 
&Q_{n+1} = \frac{\alpha}{\overline o_n}(R_n + (1-\alpha)R_{n-1}+(1-\alpha)^2R_{n-2}+ ...... + (1-\alpha)^{n-1}\overline R_1)  \\
其中 \\ 
&\overline o_n = \overline o_{n-1} + \alpha(1-\overline o_{n-1})  \\
&\overline o_n = (1-\alpha)\overline o_{n-1}+\alpha  \\
&\overline o_n = (1-\alpha)((1-\alpha)\overline o_{n-2}+\alpha)+\alpha  \\
&\overline o_n = (1-\alpha)^2\overline o_{n-2}+(1-\alpha)\alpha+\alpha  \\
&\overline o_n = (1-\alpha)^{n-1}\overline o_1 + ...... +(1-\alpha)\alpha+\alpha  \\
&\overline o_n = (1-\alpha)^{n-1}\alpha + ...... +(1-\alpha)\alpha+\alpha  \\
综上  \\
&Q_{n+1} = \frac{\sum_{i=1}^n(1-\alpha)^{n-i}R_i}{\sum_{i=1}^n(1-\alpha)^{n-i}}  \\
\end{align}  \\
$$



1.针对每一个$R_k$，它的系数为$\frac{(1-\alpha)^{n-k}}{\sum_{i=1}^n(1-\alpha)^{n-i}}$ 

$R_k$的系数依赖于它离当前时间的长短。即 exponential recency-weighted average

而且，随着$n$增长，分母变大（因为求和的数在原有的基础上增加了）。分子部分$0 \lt 1-\alpha \lt 1$,$n-k$变大，所以分子是变小。所以$R_k$的权重还是呈衰减趋势的。

2.$Q_{n+1}​$（奖励的估计值）与初始$Q_1​$无关。



---

### policy and value function

state-value function $v_{\pi}(s)$ 用来衡量在policy $\pi$ 下，当前状态的好坏程度。

执行不同的 policy ，得到不同的 $v_{\pi}(s)$，如果一个 policy 满足要求$v_{\pi}(s) \ge v_{\pi^{'}} \quad \forall s \in S$，这个$\pi$就是 optimal policy $\pi_*$，此时的 value function 记作 $v_{*}(s) = max_{\pi}v_{\pi}(s)$。

action-value function $q_{\pi}(s, a)$ 用来衡量在policy $\pi$ 下，执行动作a的好坏程度。

针对同一组(s, a)， 执行不同的 policy，得到的$q_{\pi}(s, a)$不同。同理，optimal policy $\pi_*$ 使 $q_{\pi_*}(s, a)$最大。

$q_*(s, a) = max_{\pi}q_{\pi}(s, a)$。





