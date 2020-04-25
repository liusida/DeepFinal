## Apr 24 (1)

Start two exp on Deep Green
each has 4 GPUS
seed=0
with and without DNN
They should behave the same in first 40 generation, and after that, DNN start to select. (It will happen 80 min later)

The result of first exp:
start 15:46, end 17:28
DNN 86 generation
wo_DNN 114 generation
at generation 40: both 
Gladys Virginia's fitness score: 8.2e-02 
George Mary's fitness score: 8.1e-02 
Kirk David's fitness score: 8.0e-02 

after that 
wo DNN 0.x min a generation
DNN 1 min a generation

## Apr 24 (2)

Start two exp again, with mutate arg [4, 0.3], so it will mutate more aggresively.
seed=1

It works. The boxplot of every generation clearly shows that DNN removed almost all candidates with low fitness score.

But the bests of two groups seems similar.

## Apr 24 (3)

Let try two exp again, with even higher mutate rate [1, 0.5]. (I was thinking increase the body dimension, but I don't want to redo the previous work. keep 6x6x6)
seed=2

High mutate rate [1, 0.5] seems failed. Turn back to [4, 0.3].

add another six exp in queue.
seed=3 (resubmit, in queue)
seed=4 (started)
seed=5

## Apr 25 (1)

seed=4 works
seed=5 seems no improvement
seed=3 is still running

Let's introduce a means called "Great Distinction"! population will reduce to 3 every 40 generation.

seed=6,7,8 all with three trials: DNN_ds wo_DNN_ds wo_DNN



