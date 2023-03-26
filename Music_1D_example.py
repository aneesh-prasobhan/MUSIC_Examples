import numpy as np
import matplotlib.pyplot as plt

# ======= (1) TRANSMITTED SIGNALS ======= #

# Signal source directions
az = np.array([35, 39, 127])  # Azimuths
el = np.zeros_like(az)  # Simple example: assume elevations zero
M = len(az)  # Number of sources

# Transmitted signals
L = 200  # Number of data snapshots recorded by receiver
m = np.random.randn(M, L)  # Example: normally distributed random signals

# ========= (2) RECEIVED SIGNAL ========= #

# Wavenumber vectors (in units of wavelength/2)
k = np.pi * np.column_stack((np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                             np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                             np.sin(np.deg2rad(el)))).T
# print k
print("k:")
print(k.shape)
print(k)

# Array geometry [rx, ry, rz]
N = 10  # Number of antennas
r = np.column_stack((np.arange(-(N-1)/2, (N-1)/2 + 1).T, np.zeros((N, 2))))

# Matrix of array response vectors
A = np.exp(-1j * r @ k)
#Print(A)
print("A:")
print(A.shape)
print(A)

# Additive noise
sigma2 = 0.01  # Noise variance
n = np.sqrt(sigma2) * (np.random.randn(N, L) + 1j * np.random.randn(N, L)) / np.sqrt(2)

# Received signal
x = A @ m + n

# ========= (3) MUSIC ALGORITHM ========= #

# Sample covariance matrix
Rxx = x @ x.conj().T / L

# Eigendecompose
eig_values, eig_vectors = np.linalg.eig(Rxx)
idx = np.argsort(eig_values)
E = eig_vectors[:, idx]
En = E[:, :-M]  # Noise eigenvectors (ASSUMPTION: M IS KNOWN)

# MUSIC search directions
AzSearch = np.arange(0, 181).T  # Azimuth values to search
ElSearch = np.zeros_like(AzSearch)  # Simple 1D example

# Corresponding points on array manifold to search
kSearch = np.pi * np.column_stack((np.cos(np.deg2rad(AzSearch)) * np.cos(np.deg2rad(ElSearch)),
                                   np.sin(np.deg2rad(AzSearch)) * np.cos(np.deg2rad(ElSearch)),
                                   np.sin(np.deg2rad(ElSearch)))).T
ASearch = np.exp(-1j * r @ kSearch)

# MUSIC spectrum
Z = np.sum(np.abs(ASearch.conj().T @ En) ** 2, axis=1)

# Plot
plt.plot(AzSearch, 10 * np.log10(Z))
plt.title("Simple 1D MUSIC Example")
plt.xlabel("Azimuth (degrees)")
plt.ylabel("MUSIC spectrum (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
