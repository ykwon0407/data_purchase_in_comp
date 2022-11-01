# Competition over data: how does data purchase affect users?

This repository provides a key implementation of the paper *[Competition over data: how does data purchase affect users?](https://openreview.net/forum?id=63sJsCmq6Q)* accepted at [Transactions on Machine Learning Research (TMLR)](https://www.jmlr.org/tmlr/). In this paper, we introduce **a new environment in which ML predictors use active learning algorithms to effectively acquire labeled data within their budgets while competing against each other**. We empirically show that the overall performance of a machine learning predictor improves when predictors can actively query additional labeled data. Suprisingly, however, the quality that users experience---i.e., the accuracy of the predictor selected by each user---can decrease even as the individual predictors get better.

### Quick start

```
python3 launcher.py run --exp-id 004IY --run-id 0
```

This will run the first experiment `004IY` defined in `config.py`. Here, `004IY` considers the `insurance` user distribution and 18 machine learning competitors with 400 initial budgets. By default, all ML competitors use the uncertainty-based active learning data purchase algorithm. Implementation details are available in the paper. The Python command above create `config.pickle` and `special_log` in the `results` folder. With the outputs, `notebooks/Insurance_data_purchase.ipynb` shows how to evaluate metrics used in the paper.

### Authors

- Yongchan Kwon (yk3012 (at) columbia (dot) edu)

- Tony Ginart (tginart (at) stanford (dot) edu)

- James Zou (jamesz (at) stanford (dot) edu)

### References

- P. van der Putten and M. van Someren (eds) . CoIL Challenge 2000: The Insurance Company Case.  Published by Sentient Machine Research, Amsterdam. Also a Leiden Institute of Advanced Computer Science Technical Report 2000-09. June 22, 2000. 



