
# coding: utf-8

# # A Structural Risk Minimization Approach to Principal Component Analysis
# 
# ## by 
# 
# ## Fayyaz Minhas

# The objective of this tutorial is to provide a fundamental (though somewhat simplistic) description of principal component analysis in a "learn-by-doing" style approach in which you are asked questions that can be answered by playing with the code included in the tutorial.
# 
# How can we find redundant or dimensions with little or no information in a given data? Remember, that for a given variable, the amount of information in it is proportional to its variance -- if all data is constant, then its variance is zero and so is its information content. If we have very high dimensional data, we can reduce its dimensionality by projecting it along directions (or vectors) such that the variance along the chosen direction is maximized in order to preserve the most information in the data. For example, consider two (related) variables: height and weight of all individuals in a class. If we can predict the weight from the  height then we do not need to store the weight dimension for all data points. However, without such a prediction system in place, or, if there is no relationship between weight and height, it will be impossible to reconstruct weight from height and the information of weight will be lost for ever if the weight dimension is dropped. On the other hand, if the weight of an individual is linearly dependent to his or her height, i.e., $w \approx ah+b$, then we can fit a simple line to get one from the other by reducing the error between the true value of weight and the value predicted on the basis of height (called the reconstruction error). If the reconstruction error is low, we can simply store the linear predictor $(a,b)$ and the height values and throw away the weight without much loss in information. 
# 
# Principal component analysis is a method for finding orthogonal directions of maximum variance in the data such that if the data is projected in those directions, the variance of the projected data is maximum. The dimensionality of the data can be reduced by projecting it along those directions. This projection along the direction of maximum variance gives minimum information loss. Below, we discuss how finding the direction of maximum variance for a given data set corresponds to finding the eigen vector of the covariance matrix of the data and how it leads to a minimum loss in information if the data is projected in that direction. 
# 
# Let's assume that we are given $N$ $d$-dimensional data points $\mathbf x_i, i=1...N$. We want to find the direction vector $\mathbf w$ such that the projection $z_i=\mathbf w^T \mathbf x_i$ for a point $\mathbf x_i$ has maximum variance. 
# 
# To find the direction of maximum variance, we will need to develop a mathematical formula that can help us calculate the variance of the data after it has been projected along a certain direction so that we can search for the direction of maximum variance by optimizing over that formula. Let's calculate the variance after projection of the data along $\mathbf w$. We know that variance of the values $z_i, i=1...N$ is the expected (average) value of the squared deviation $(z_i-\mu_z)$ around the mean value $\mu_z = \frac{1}{N}\sum_{i=1}^{N}z_i=\frac{1}{N}\sum_{i=1}^{N}\mathbf w^T\mathbf x_i=\mathbf w^T\frac{1}{N}\sum_{i=1}^{N}\mathbf x_i=\mathbf w^T\mu_x$ where $\mathbf{\mu_x}=\frac{1}{N}\sum_{i=1}^{N}\mathbf x_i$ is the $d$-dimensional vector of average values of all data points along each of the $d$ data dimensions. Therefore,
# 
# $var(z)=var(\mathbf w^T \mathbf x)=\frac{1}{N}\sum_{i=1}^{N}[(\mathbf w^T \mathbf x_i-\mathbf w^T\mathbf\mu_x)^2]$
# 
# $=\frac{1}{N}\sum_{i=1}^{N}[(\mathbf w^T\mathbf x_i-\mathbf w^T\mathbf\mu_x)(\mathbf w^T\mathbf x_i-\mathbf w^T\mathbf\mu_x)]$
# 
# $=\frac{1}{N}\sum_{i=1}^{N}[(\mathbf w^T\mathbf x_i-\mathbf w^T\mathbf\mu_x)(\mathbf x_i^T\mathbf w-\mathbf \mu_x^T \mathbf w)]$
# 
# $=\frac{1}{N}\sum_{i=1}^{N}[\mathbf w^T(\mathbf x_i-\mathbf \mu_x)(\mathbf x_i-\mathbf \mu_x)^T\mathbf w]$
# 
# $=\mathbf w^T\frac{1}{N}\sum_{i=1}^{N}[(\mathbf x_i-\mathbf \mu_x)(\mathbf x_i-\mathbf \mu_x)^T]\mathbf w$
# 
# $=\mathbf w^T\mathbf C \mathbf w$
# 
# Here, $\mathbf C=\frac{1}{N}\sum_{i=1}^{N}[(\mathbf x_i-\mathbf \mu_x)(\mathbf x_i-\mathbf \mu_x)^T]$ is the $d \times d$ sized "covariance matrix". The covariance of two variable $a$ and $b$ over $N$ values $a_i,b_i, i=1...N$ is given by $c(a,b) = \frac{1}{N}\sum_{i=1}^{N}[(a_i-\mu_a)(b_i-\mu_b)^T]$ is a measure of the linear relationship between two variables - in our case two dimensions of $\mathbf x$ or two of our features. Covariance will be high (positive) if increase in values of one variable above its mean correlates with increase in values of the other variable above the other variable's mean value. Covariance will be high (negative) if increase in values of one variable above its mean correlates with decrease in values of the other variable below the other variable's mean value. Covariance will be low (small positive or negative) if increase in values of one variable above its mean has little effect or little correlation with increase in values of the other variable above the other variable's mean value. In our example below, we take two variables, height and weight of a person which are expected to be correlated and exhibit high covariance. It is interesting to note that if we take a single variable $b=a$ then variance becomes covariance. A covariance matrix of $d$ variables is a $d \times d$ matrix of all pairwise covariances. Note that the covariance matrix will be symmetric since $cov(a,b)=cov(b,a)$.  It is typically better to scale the variables to the same range prior to covariance calculation to reduce effects of differences in range of values from affecting the covariance value. This can be achieved by subtracting the value of a variable from its mean and dividing by its standard deviation in a process called mean-standard deviation normalization (or standardization).
# 
# Uptil now we have expressed the variance of the projected data in terms of $\mathbf w$ used for projection and the covariance matrix of the data which can be calculated beforehand. Remember that we are interested in finding the direction of maximum variance as the projection of the given data along this direction entails minimum loss in information. Now using the princple of structural risk minimization, the machine learning problem of finding the optimal direction vector $w$ that minimize the loss in information after projection while ensuring regularization can be written as:
# 
# $min_\mathbf w \alpha R(\mathbf w)+E(\mathbf w)$
# 
# Where $E$ is the error term and $R$ controls the (inverse of) regularization. Since error is inversely related to variance of $z$, we can write this learning problem as follows:
# 
# $min_\mathbf w \alpha ||\mathbf w||^2-var(z)$ (for $\alpha \ge 0$)
# 
# or
# 
# $min_\mathbf w \alpha \mathbf w^T\mathbf w -\mathbf w^TC\mathbf w$
# 
# Here, we want to minimize the value of the norm of $\mathbf w$ while maximizing the variance. 
# 
# Taking the derivative with respect to $\mathbf w^T$ and substituting to zero, we get:
# 
# $\mathbf C\mathbf w=\alpha \mathbf w$
# 
# Note that there is a trivial solution to this $\mathbf w = \mathbf 0$. However, by constraining $||\mathbf w||=1$, we can find a non-trivial solution. For this purpose, note that the above equation is an Eigen value problem with eigen vector $\mathbf w$ and eigen value $\alpha$. An Eigen vector $\mathbf w$ is a property of a matrix $\mathbf C$ such that the vector resulting from matrix multiplication $\mathbf C\mathbf w$ is in the same direction as $\mathbf w$ with only scaling by a corresponding constant factor $\alpha$ called Eigen value. The number of Eigen values and vectors is equal to the number of dimensions of the matrix. These vectors of a covriance matrix are called principal components as they correspond to directions of maximal variance. Hence the name "Principal Component Analysis".
# 
# Thus, the direction of maximum variance $\mathbf w$ corresponds to the Eigen vector of the covariance matrix $\mathbf C$. Thus, if we find the Eigen vectors of the covariance matrix, we can get what we want! Below, we also discuss a simpler, though more computationally intensive, approach to finding the direction of maximum variance using a simple loop.
# 
# To gain further understanding of the concepts discussed so far, let's generate some data.

# In[52]:


import numpy as np
weights = np.array([55,74,58,52,58,67,80,71,62,69,72,67,78])
heights = np.array([56, 73, 60, 54, 56, 62, 77, 73, 72, 69, 74, 67, 84,])


# And plot it!

# In[53]:


import matplotlib.pyplot as plt
plt.scatter(heights,weights)
plt.xlabel('height')
plt.ylabel('weight')
plt.axis('square')
plt.grid()
plt.show()


# Let's make the data into a numpy array and get the mean and standard deviation of the data so we can normalize it to have zero mean and unit standard deviation (variance) along each dimension.

# In[54]:


X = np.vstack((heights,weights)).T
#X = np.random.randn(100,2)
#X[:,1]+=0.5*X[:,0]
N,d = X.shape
print("The dimensions of X are",X.shape)
Xm = np.mean(X,axis=0)
Xs = np.std(X,axis=0)
print("The mean is",Xm)
print("The standard deviation is",Xs)


# In[55]:


Xn = (X-Xm)/(Xs)
print("Mean after normalization",np.mean(Xn,axis=0))
print("Standard deviation after normalization", np.std(Xn,axis=0))
print("Variance after normalization", np.var(Xn,axis=0))
print("Total Variance after normalization", np.sum(np.var(Xn,axis=0)))


# Note that the total variance along both directions is 2.0. Let's plot the data. Notice that the trend of the data is the same whereas the mean has been changed to zero and standard deviation changed to one which is equivalent to a shifting and scaling of the data.

# In[56]:


plt.scatter(Xn[:,0],Xn[:,1])
plt.grid()
plt.xlabel('height')
plt.ylabel('weight')
plt.show()


# Let's calculate the Covariance matrix. Note that the covariance matrix is symmetric and positive semi-definite (has non-negative eigen values). In the first reading of the tutorial, ignore the code below that corresponds to the snapshot variable being True.

# In[57]:


snapshot = False #See details on the Snapshot method below 

if not snapshot:
    C = np.cov(Xn.T) # Determine dxd sized Cov. Matrix
else:
    C = np.cov(Xn) # Determine NxN sized Cov. Matrix

print("Covariance matrix Dimensionality is: ",C.shape)
print("Covariance matrix is\n",C)


# The diagonal component of the $d \times d$ covariance matrix correspond to variances of the two features we have (which are 1.0 after standardization) and the cross-diagnonal elements correspond to the covariance between our two features.  Note that the covariance is pretty high as is evident from the scatter plot as well. As a consequence, it should be possible to reduce the two dimensions to a single one with minimum loss of information. 
# 
# Let's calculate the eigen values and principal components. Once a principal component $\mathbf w$ has been calculated, we can project our data along it by $z = \mathbf w^T \mathbf x$ or in matrix form $\mathbf X_{n_{(N \times d)}}\mathbf W_{(d \times d)}$, where, $\mathbf W$ is the matrix of principal components.  We can then sort the principal components in descending order with respect to the amount of variance captured along those principle components. We will also plot the scree graph which plots the fraction of variance captured along each dimension.

# In[58]:


ev,pc = np.linalg.eig(C)
ev = np.abs(ev)

if snapshot: #ignore on the first reading
    pc = np.dot(Xn.T,pc[:,:d])
    pc/= np.linalg.norm(pc,axis=0)
    pc = np.real(pc)
cvar =  np.var(np.dot(Xn,pc),axis=0)
idx = np.argsort(-cvar)
cvar = cvar[idx]
pc = pc[:,idx]

print("The eigen vectors (principal components) are \n",pc)
print("The variance captured along each PC:",cvar)
print("The fraction of variance captured along each PC: ",np.cumsum(cvar)/np.sum(cvar))
if not snapshot:
    ev = ev[idx]
    print("The eigen values are: ",ev)
    print("The fraction of eigen values along each PC: ",np.cumsum(ev)/np.sum(ev))

plt.scatter(Xn[:,0],Xn[:,1])
plt.arrow(0,0,pc[0,0],pc[1,0],color='k',head_width=0.2)
plt.arrow(0,0,pc[0,1],pc[1,1],color='b',head_width=0.1)
plt.axis('square')
plt.xlabel('height')
plt.ylabel('weight')
plt.title("Principal Components")
plt.grid()
plt.show()

Z = np.dot(Xn,pc)
plt.scatter(Z[:,0],Z[:,1])
plt.axis('square')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.title("Data in PC Space")
plt.grid()
plt.show()
#plotting the scree plot
plt.plot(np.arange(len(cvar))+1,np.cumsum(cvar)/np.sum(cvar),'o-')
plt.axis([1,len(ev),0,1])
plt.xlabel("number of components k")
plt.ylabel("Fraction of variance @  k")
plt.grid()
plt.title("Scree plot")
plt.show()


# The above plot shows the directions of the principal components: Note that:
# 
# 1. There are two principal components: The one with the largest variance (eigen value) is called the first principal component whereas the other one is called the second principal component. 
# 2. The variance along the first principal component is higher in comparison to the second.
# 3. The variance along the first projected direction is higher than the variance along original features which is 1.0 after normalization. Thus, the principal component is a direction that captures more information than any of the original features alone.
# 4. The norm of each of the principal components is 1.0.
# 5. The two principal components are orthogonal to each other. 
# 6. As a consequence of 2-4 above, the principle component matrix and its transpose are inverses of each other, i.e., $\mathbf {W}^{-1}=\mathbf {W}^{T}$ or $\mathbf {W}^{T}\mathbf W=\mathbf I$.
# 7. The eigen values correspond to the amount of captured variance: The fraction of variance captured along a direction is exactly equal to the fraction of eigen values. Thus, the first principal component corresponds to the largest eigen value and so on.
# 8. The plot of the fraction of captured variance upto $k$ principal components (called the scree plot) can be used to select how many principal components to retain when reducing dimensionality. For the original data used in this example, upto 95% variance is along the first principal component. Therefore, if the second principal component is dropped, the loss of information will be only ~5%.

# In[59]:


pc1 = pc[:,0]
pc2 = pc[:,1]
print('Dot product of the first two principal components:',np.dot(pc1,pc2))
print('Norm of the first two principal components:',np.linalg.norm(pc1),np.linalg.norm(pc1))

print("The principle component matrix multiple by its transpose:\n",np.dot(pc.T,pc))


# Let's reduce the dimensionlity to "$d_r$" dimensions. Note that the projection is $ z = \mathbf w_{{(d \times 1)}}^T\mathbf x_{(d \times 1)}$ which, can be written as the matrix operation: $\mathbf Z_{(N \times d_r)} = \mathbf X_{{(N \times d)}}\mathbf W_{(d \times d_r)}$

# In[60]:


dr = 1
W = pc[:,0:dr] #selecting upto dr principal components only
Z = np.dot(Xn,W)
print("Data after transformation\n",Z)
print("Standard deviation after transformation",np.std(Z,axis=0))
print("Variance after transformation",np.var(Z,axis=0))
print("Fraction of variance captured along the projections: ",np.var(Z,axis=0)/np.sum(np.var(Xn,axis=0)))


# Let's calculate the direction of maximum variance using a simple for-loop-based search to verify that we have actually found the correct answer. We will generate unit vectors along a unit circle and calculate the variance of the data after projection along a given vector. We show the scatter plot of the data overlayed by projections vectors whose length has been set equal to the variance of the data projected in that direction. We will also plot the variance vs. the angle/direction of the unit vector. Note that the highest standard deviation corresponds to the first eigen vectors.

# In[61]:


import numpy as np
theta = (2*np.linspace(0,1,100)-1)*np.pi
vx = np.zeros(theta.shape[0])
maxv = 0
best_w = None
best_theta = 0
for i,t in enumerate(theta):
    wt = [np.cos(t),np.sin(t)]
    vx[i]=np.var(np.dot(wt,Xn.T))
    plt.arrow(0,0,vx[i]*wt[0],vx[i]*wt[1],color='b',head_width=0.05,linestyle=':')
    if vx[i]>maxv:
        maxv = vx[i]
        best_w = wt
        best_theta = t
plt.scatter(Xn[:,0],Xn[:,1],color='k')
plt.arrow(0,0,maxv*best_w[0],maxv*best_w[1],color='k',head_width=0.1)
plt.arrow(0,0,cvar[0]*pc[0,0],cvar[0]*pc[1,0],color='r',head_width=0.1,linestyle=':')

plt.grid()
plt.xlabel('height')
plt.ylabel('weight')
plt.axis('square')
plt.title("Variance in Various Directions")
plt.show()


plt.plot(theta*180/np.pi,vx)
plt.scatter(best_theta*180/np.pi,maxv,color = 'k')
plt.xlabel("theta")
plt.ylabel("variance")
plt.grid()
plt.show()
print("Maximum Variance:",maxv)
print("Direction of Maximum Variance:",best_w)



# Now, Let's calculate the inverse projection. Note that, the projections can be written as $\mathbf z_{(d \times 1)} = \mathbf W_{(d \times d)}^T \mathbf x_{(d \times 1)}$, therefore, the inverse projection (re-projection) can be written as $\mathbf x^r_{(d \times 1)} = \mathbf {W}^{-1} \mathbf z_{(d \times 1)}$. We already know that $\mathbf {W}^{-1} = \mathbf {W}^{T}$. We can use this to calculate the inverse transform using $\mathbf x^r_{(d \times 1)} = \mathbf {W}^{-1} \mathbf z_{(d_r \times 1)} = \mathbf {W}^{T}_{(d \times d_r)} \mathbf z_{(d_r \times 1)} $. In matrix form we have $\mathbf X^r = \mathbf Z \mathbf W^T=\mathbf X \mathbf W \mathbf W^T$. We will de-normalize the data by multiplying the vector of standard deviations and adding the mean vector. We then plot the re-projected points.

# In[62]:


iW = W.T
Xr = np.dot(Z,iW)*Xs+Xm
plt.scatter(X[:,0],X[:,1],marker = '+')
plt.scatter(Xr[:,0],Xr[:,1])
plt.legend(['Original','Reprojection'])
for x,xr in zip(X,Xr):    
    plt.arrow(x[0],x[1],xr[0]-x[0],xr[1]-x[1])
plt.axis('square')
plt.xlabel('height')
plt.ylabel('weight')
plt.grid()
plt.show()


# The reprojection or reconstruction error over the normalized data can be calculated as: $\sum_{i=1}^{N}\left\lVert \mathbf x_i - \mathbf x^r_i \right\rVert^2=\left\lVert \mathbf X - \mathbf X^r \right\rVert_F^2$, where $\mathbf x^r = \mathbf {W}^{-1} \mathbf z = \mathbf {W}^{T} \mathbf z $ and $\mathbf X^r=\mathbf Z \mathbf W^T$. 
# 
# As shown below, the average reconstruction error is equal to the difference between total variance and variance along the direction of projection.
# 
# Thus, the direction of maximum variance is the direction of minimum information loss as well as the minimum reconstruction error. Therefore, the objective function of PCA can also be written as: 
# 
# $min_{\mathbf W}\left\lVert \mathbf X - \mathbf X \mathbf W \mathbf W^T \right\rVert_F^2$ 
# 
# subject to the constraint
# 
# $\mathbf W^T \mathbf W = \mathbf I$.
# 

# In[63]:


print("Average Reconstruction Error over normalized data: ",np.mean(np.linalg.norm(np.dot(Z,iW)-Xn,axis=1)**2))
print("Difference between total variance and variance along projection:",np.sum(np.var(Xn,axis=0))-np.sum(np.var(Z,axis=0)))


# As discussed earlier, the reconstruction error is equal to the variance that is lost due to projection. In other words, PCA can be interpreted as: 1) Projecting the data along the direction of maximum variance, or equivalently, 2) Projecting the data along the direction in which re-projection error is minimized. Thus, the loss function for PCA is the re-projection error which is equivalent to negative of the amount of captured variance.
# 
# That's all folks! 
# 
# ### Snapshot Method
# 
# Note that the covariance matrix $\mathbf C=\frac{1}{N}\sum_{i=1}^{N}[(\mathbf x_i-\mathbf \mu_x)(\mathbf x_i-\mathbf \mu_x)^T]\equiv \mathbf X_n \mathbf X_n^T$ calculated above is $d \times d$. Here, $\mathbf X_n$ is the normalized data matrix of size $d \times N$. However, if $d>N$, as in the case of image data, it would be computationally more efficient to formulate an equivalent covariance matrix $\mathbf C'\equiv \mathbf X_n^T \mathbf X_n$ of size $N \times N$ and find its $N \times 1$ eigen vectors $\mathbf w'$ which can then be used to compute the $d \times 1$ eigen vectors $\mathbf w$ of $\mathbf C$ by $\mathbf w = \mathbf X_n^T \mathbf w'$ and normalizing $\mathbf w$ to unit-norm. This is called the snapshot method and it is widely used for cases in which $d>N$. You can test this method out by setting the variable 'snapshot' to True in the above code. 
# 
# ### Exercises
# 
# Here is what you can do:
# 
# 1. Change the data and see its impact. You can change it by adding some gaussian noise using the np.random.randn function. See how this affects the scatter plot, the covariance matrix, the scree plot and the principal components and their Eigen values.
# 
# 2. Can you overlay the principal components of your data as vectors over the scatter plot to see whether the direction of maxium variance being found is actually correct. 
# 
# 3. Try changing the number of reduced dimensions from 1 to 2. What happens? Please explain your observations.
# 
# 4. See how can you apply PCA from the library scikit-learn and use it for classification or regression coupled with an appropriate machine learning model. It is important to note that PCA is an unsupervised technique and labels or target values should not be used while determining the principal components.
# 
# 5. How do we apply PCA over previously unseen data?
# 
# 6. How can PCA be used for visualization of high dimension data?
# 
# 7. How is PCA used in Eigen Faces for face recognition?
# 
# 8. How can PCA be used for clustering (optional).
# 
# 9. What are the underlying assumptions of PCA? (optional)
# 
# 10. What are the limitations of PCA? (optional)
# 
# 11. How can PCA be kernelized? (optional)
# 
# 12. What happens if you project the data onto the second principal component only? (optional)
# 
# 13. If the number of features is large, then the covariance matrix is going to be huge. How can we apply PCA to high dimensional data? (Hint: Snapshot method for PCA)
# 
# 14. How can PCA be used for data transmission?

# ### Application: Eigen Digits
# 
# Let's apply PCA to the MNIST dataset which consists of 8 by 8 images of numbers. We can use the scikit-learn PCA tool. This approach is very similar to Eigen faces which are used for face recognition. We can call this "Eigen Digits".

# Let's load the data using sklearn.

# In[64]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# Let's apply PCA to this data. We will keep the number of principal components equal to the number of dimensions (64) to see how much variance we can capture with increasing the number of components. This can be done using the scree graph which plots the proportion of captured variance vs. the number of components being used.
# 
# How many components are required for capturing 90% variance?
# 
# How many components are required for near perfect reconstruction?

# In[65]:


from sklearn.decomposition import PCA #import PCA
pca64 = PCA(n_components=64)
pca64.fit(digits.data) #training PCA
projected = pca64.transform(digits.data) #projecting the data onto Principal components
print(digits.data.shape)
print(projected.shape)
plt.plot(np.arange(len(pca64.explained_variance_ratio_))+1,np.cumsum(pca64.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca64.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()


# Let's show the original data and the projected data as images. We take two arbitrary digits and see the effect of applying PCA. Note that the projected digits look nothing like the original ones but are definitely different from each other. Note that the projected digits have a large number of dimensions equal to zero.

# In[66]:


plt.imshow(digits.data[0,:].reshape(8,8),cmap='gray'); plt.title('Original'); plt.colorbar(); plt.show()
plt.imshow(projected[0,:].reshape(8,8),cmap='gray'); plt.title('After PCA');  plt.colorbar(); plt.show()
plt.imshow(digits.data[900,:].reshape(8,8),cmap='gray'); plt.title('Original'); plt.colorbar(); plt.show()
plt.imshow(projected[900,:].reshape(8,8),cmap='gray'); plt.title('After PCA');  plt.colorbar();plt.show()


# ### PCA for visualization
# 
# PCA can be used for visualization. Let's plot two principal components against each other. This allows us to visualize 64 dimensional data in two dimensions for exploratory data analysis. As you can see below, you get a pretty good clustering in the PCA space in which similar examples corresponding to the same digit cluster together. 
# 
# What digits are more similar to each other? 
# 
# What happens when you change the principal components for visualization? 
# 
# Can you find the principal components that provide the best separation between classes?

# In[67]:


i1 = 0 #first principal component
i2 = 1 #second principal component
plt.scatter(projected[:, i1], projected[:, i2],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10));
plt.grid()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar();
plt.show()


# Note that the principal components are 64 dimensional vectors which can be viewed as digits. Below, we show the two principal components selected above and the principal component correspondng to lowest variance. These images show us what kind of patterns are being learned. Notice that the outer boundary is zero for all principal components because no digits occur there so the image can easily be reduced in dimensions.

# In[68]:


plt.imshow(pca64.components_[i1,:].reshape(8,8),cmap='gray'); plt.title('PC1'); plt.colorbar(); plt.show()
plt.imshow(pca64.components_[i2,:].reshape(8,8),cmap='gray'); plt.title('PC2'); plt.colorbar(); plt.show()
plt.imshow(pca64.components_[-1,:].reshape(8,8),cmap='gray'); plt.title('Last PC'); plt.colorbar(); plt.show()


# ### PCA Reconstruction
# Now we will reduce the image dimensions from $(d,d)$ to $(d_r,d_r)$ by picking the dimensions using PCA. Note that this is different from resizing the image which can incur a large information loss in comparison to PCA. So a (3,3) image of 9 dimensions would allow near perfect reconstruction to the original image but a (64,64) to (3,3) down-scaling can severly affect the image. Below we show the original, transformed and reconstructed image. 
# 
# Try it for different digits.
# 
# See what is the effect of changing the number of dimensions or reconstruction accuracy? 

# In[69]:


dr = 4
pca = PCA(n_components=dr*dr)
components = pca.fit_transform(digits.data)
reconstruction = pca.inverse_transform(components)
plt.imshow(digits.data[0,:].reshape(8,8),cmap='gray',vmin=0,vmax=20); plt.title('Original'); plt.colorbar(); plt.show()
plt.imshow(components[0,:].reshape(dr,dr),cmap='gray'); plt.title('Projection'); plt.colorbar(); plt.show()
plt.imshow(reconstruction[0,:].reshape(8,8),cmap='gray',vmin=0,vmax=20); plt.title('Reconstruction'); plt.colorbar(); plt.show()


# (c) Dr. Fayyaz Minhas *afsar at pieas dot edu.pk#. [http://faculty.pieas.edu.pk/fayyaz]
