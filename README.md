# Principle-Component-Analysis

Implementing a facial analysis program using PCA, using the skills like linear algebra + PCA. 

1. load_and_center_dataset(filename): load the dataset from the provided .npy file, center it around
the origin, and return it as a numpy array of floats.

2. get_covariance(dataset): calculate and return the covariance matrix of the dataset as a numpy
matrix (d × d array).

3. get_eig(S, m): perform eigendecomposition on the covariance matrix S and return a diagonal matrix (numpy array) with the largest m eigenvalues on the diagonal in descending order, and a matrix (numpy array) with the corresponding eigenvectors as columns.

4. get_eig_prop(S, prop): similar to get_eig, but instead of returning the first m, return all eigen- values and corresponding eigenvectors in a similar format that explain more than a prop proportion of the variance (specifically, please make sure the eigenvalues are returned in descending order).

5. project_image(image, U): project each d × 1 image into your m-dimensional subspace (spanned by m vectors of size d × 1) and return the new representation as a d × 1 numpy array.

6. display_image(orig, proj): use matplotlib to display a visual representation of the original image and the projected image side-by-side.
