# Wgan-GP_cats

This is an implementation of Wasserstein GANs with gradient penalty.<br>
Link to the paper is : https://arxiv.org/pdf/1704.00028.pdf
<br><br><br>

Wasserstein GANs use the Earth mover distance instead of TV or JS divergence or KL divergence.
<br>
The weaker the distance, the better is the convergence of GANs. <br>
The other distances mentioned failed in the case of low dimensional manifolds where the distributions may have very little common projection space. 
<br>
The mathematical details of the advantages of this distance can be read here : https://arxiv.org/pdf/1701.07875.pdf<br>
<br>
<br>

## Results
### Epoch 1 
<img src="sample_images/wgan_gp/Epoch 1.jpg">
<br>
### Epoch 100
<img src="sample_images/wgan_gp/Epoch 100.jpg">
<br>
### Epoch 300
<img src="sample_images/wgan_gp/Epoch 300.jpg">
<br>
### Epoch 500
<img src="sample_images/wgan_gp/Epoch 500.jpg">
