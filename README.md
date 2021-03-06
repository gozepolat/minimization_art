# Perona-Malik with PyTorch and CNNs
Perona-Malik diffusion is a well known regularization technique that preserves edges. Although we normally use a constant kernel to calculate the gradients (i.e. an edge detector such as Laplacian), using CNNs and PyTorch, it is possible to explore operators other than edge detectors. When combined with an appropriate loss function and the right diffusion rate vs learning rate, this allows exploring aesthetically pleasing diffusion reaction transformations which still temporarily preserve edges to some extent. Feel free to see my [blog](https://gozepolat.github.io/perona_malik_on_drugs) for further details!

---

<p align="center">
   <img src="images/profile.gif?raw=True">
</p>


