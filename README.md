<p align="left">
  <img width="460" height="300" src="images/star.gif?raw=True">
</p>
Perona-Malik diffusion is a well known regularization technique that preserves edges. Although we normally use a constant kernel to calculate the gradients (i.e. an edge detector such as Laplacian), using CNNs and PyTorch, it is possible to explore operators other than edge detectors. When combined with an appropriate loss function and the right diffusion rate vs learning rate, this allows exploring edge preserving and aesthetically pleasing transformations.
--

