# Physical Understanding of the Project

We have a robot with camera so a moving camera. We are in a world full of id-ed landmarks, yet we don’t have the map of the world. At each observation we are receiving a 2D image from the camera that contains some of the id-d landmarks.

We will use epipolar geometry to initialize our algorithm and map.
Then we will use projective-ICP to perform odometry, while keep using epipolar geometry for the initial placement of the newly incoming landmark observations.

# Detailed Theoretical Procedure followed in the Project

## Initialization

First two image observations, image 0 & image 1, being taken,

→ If we have less then 8 correspondences give error. Because we will use 8 point algorithm.

→ Find the correspondences of the landmarked in the images, that is find the landmarks that are appeared in both of the images. 

→ Use the camera frame of image 0 as the world frame.

For every matched landmark in the images, recover the bearings $d$ of the cameras from the images by undoing the camera matrix, and normilize. 

$$z_0 = \begin{Bmatrix}u_0 \\\\ v_0 \\\\ 1\end{Bmatrix}$$ and $$z_1 = \begin{Bmatrix}u_1 \\\\ v_1 \\\\ 1\end{Bmatrix}$$ undo,  $$\begin{matrix}\bar{d}_0 =\\\\ \bar{d}_1 = \end{matrix} \begin{matrix}K^{-1}z_0 \\\\ K^{-1}z_1\end{matrix}$$ normalize,  $$d = \frac{\bar{d}}{\vert\vert\bar{d}\vert\vert}$$

Then use the condition of intersection $d_1^TEd_0=0$ to calculate the matrix $E$. For this first build the matrix $A$ for each correspondence using the bearings, then find $H$ and calculate its smallest eigenvalue and the corresponding eigenvector, recover $E$ from the eigenvalue.

$A_i = (d^x_1d^x_0\\:\\:d^x_1d^y_0\\:\\:d^x_1d^z_0\\:\\:d^y_1d^x_0\\:\\:d^y_1d^y_0\\:\\:d^y_1d^z_0\\:\\:d^z_1d^x_0\\:\\:d^z_1d^y_0\\:\\:d^z_1d^z_0)$ “for the i-th landmark”

$H = \sum_i A_i^T A_i$ → let’s $e$ be the eigenvector corresponding to smallest eigenvalue.

Finally recover $E$ from vector $e$.

Now we can compute the transformation matrix from the world frame to the first camera frame using the matrix $E$.

Let’s define the SVD decomposition of $E = USV^T$ . Then, $R = UWV^T$  or  $R = UW^TV^T$ and, $t_{skew}=VSW^TV^T$  or $-t_{skew}=VSW^TV^T$ 

where, 

$$W = \begin{Bmatrix} 0 & 1 & 0 \\\\ -1 & 0 & 0 \\\\ 0 & 0 & 1 \end{Bmatrix}$$

Notice, we have four possible pairs for $R$ & $t$. In order to be able to choose the suitable pair we will test each of them and find the one that result in positive sclaing values for both camera instant since the camera line must be in front of the camera. For one matched landmark point on the image let’s call it $p$,

$p=s_0d_0=t+Rd_1s_1$

$(d_0\\:\\:-Rd_1)(s_0\\:\\:s_1)^T=t$  

$(s_0\\:\\:s_1)^T=[(d_0\\:\\:-Rd_1)^T(d_0\\:\\:-Rd_1)]^{-1}(d_0\\:\\:-Rd_1)^Tt$

⇒ Choose the pair that result in $s_0 \geq 0$ and $s_1 \geq 0$

Moreover, note that we recover only the direction of $t$ not it’s magnitude,

$t \sim \alpha t$ → so we can set at initiliziation $\vert\vert t \vert\vert=1$

Now we have the positions of the landmarks w.r.t to the world frame using triangulation, for each landmark i,

$p^i_{world}=\frac{1}{2}(s_0d_0+t+s_1Rd_1)$ 

Where we took the average from the image 0 and 1 as a robustness measure against noise.

Of course we also have trajectory from instant 0 to instant 1,

$$X^{cam}_{world}=\begin{pmatrix}R&t\\\\0&1\end{pmatrix}$$
