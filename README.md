# An example of MHL
This is an example of mhl.

In this paper, we continue to follow the magnitude pruning method and perform data augmentation on the data, and the specific techniques are shown as follows:

\begin{enumerate}
    \item Initially, encoders $h_{MM}$ and $h_{G}$, are randomly initialized with parameters respectively.
    \item During each iteration $l$, prune $e\%$ of the smallest-magnitude parameters in $\theta_G^l$ and $\theta_T^l$ by creating masks $m_{MM}^l$ and $m_{G}^l$.
    \item Apply these masks to the feed-forward encoders $f_G$ and $f_{MM}$, and compute the contrast loss.
    \item Remove the masks from the pruned parameters and update the encoders' parameters by optimizing the contrast loss.
\end{enumerate}
