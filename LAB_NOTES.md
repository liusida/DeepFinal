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

Let try two exp again, with even higher mutate rate, and 8x8x8 body (larger search space).

