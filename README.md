# Wgan-GP_cats

This is an implementation of Wasserstein GANs with gradient penalty.<br>
Link to the paper is : https://arxiv.org/pdf/1704.00028.pdf
<br><br><br>

Wasserstein GANs use the Earth mover distance instead of TV or JS divergence or KL divergence.
<br>
The weaker the distance, the better is the convergence of GANs. <br>
The other distances mentioned failed in the case of low dimensional manifolds where the distributions may have very little or null Borel Set.
<br>
The mathematical details of the advantages of this distance can be read here : https://arxiv.org/pdf/1701.07875.pdf<br>
