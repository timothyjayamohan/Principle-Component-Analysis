from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    data = np.load(filename)
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    return centered_data

def get_covariance(dataset):
    # Your implementation goes here!
    n, d = dataset.shape
    return (1 / (n - 1)) * np.dot(np.transpose(dataset), dataset)

def get_eig(S, m):
    # Your implementation goes here!
    eigenvalues, eigenvectors = eigh(S)
    # Sort the eigenvalues and vectors in decreasing order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices][:m]
    eigenvectors = eigenvectors[:, sorted_indices][:, :m]
    return np.diag(eigenvalues), eigenvectors



def get_eig_prop(S, prop):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(S)
    
    # Sort them in descending order based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate the proportion of variance explained by each eigenvalue
    total_variance = np.sum(eigenvalues)
    proportion_variance = eigenvalues / total_variance
    
    # Find indices where the proportion of variance explained by eigenvalues is greater than prop
    indices_to_keep = np.where(proportion_variance >= prop)[0]
    
    # Subset the eigenvalues and eigenvectors using the indices
    eigenvalues = eigenvalues[indices_to_keep]
    eigenvectors = eigenvectors[:, indices_to_keep]
    
    return np.diag(eigenvalues), eigenvectors


def project_image(image, U):
    # Your implementation goes here!
    # Projecting the image to the eigen space
    alpha = np.dot(U.T, image)
    
    # Reconstructing the image
    projection = np.dot(U, alpha)
    return projection



def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    #return fig, ax1, ax2
    
     # 1. Reshape the images to 32x32
    orig = orig.reshape(32, 32)
    proj = proj.reshape(32, 32)
    
    orig = np.rot90(orig.reshape(32, 32), k=-1)
    proj = np.rot90(proj.reshape(32, 32), k=-1)
    # 2. Create subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    
    # 3. Title the subplots
    ax1.set_title("Original")
    ax2.set_title("Projection")
    
    # 4. Display images
    orig_im = ax1.imshow(orig, aspect='equal')
    proj_im = ax2.imshow(proj, aspect='equal')
    
    # 5. Add colorbars
    fig.colorbar(orig_im, ax=ax1, orientation='vertical')
    fig.colorbar(proj_im, ax=ax2, orientation='vertical')
    
    # 6. Turn off the axis labels
    ax1.axis('off')
    ax2.axis('off')
    
    # 7. Return fig, ax1, and ax2
    return fig, ax1, ax2


x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
fig, ax1, ax2 = display_image(x[0], projection)