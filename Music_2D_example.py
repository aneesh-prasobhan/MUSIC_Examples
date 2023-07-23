import numpy as np
import matplotlib.pyplot as plt

# ======= (1) TRANSMITTED SIGNALS ======= #

# Signal source directions
az = np.array([35, 39, 127])  # Azimuths
el = np.array([63, 20, 57])  # Elevations
M = len(az)  # Number of sources

# Transmitted signals
L = 200  # Number of data snapshots recorded by receiver
m = np.random.randn(M, L)  # Example: normally distributed random signals

# ========= (2) RECEIVED SIGNAL ========= #

# Wavenumber vectors (in units of wavelength/2)
k = np.pi * np.column_stack((np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                             np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
                             np.sin(np.deg2rad(el)))).T



# Array geometry [rx,ry,rz] (example: 4x4 square array)
N = 16  # Number of antennas
array_size = int(np.sqrt(N))
print(array_size)
dx = 1  # distance between antennas in x direction
dy = 1  # distance between antennas in y direction
rx, ry = np.meshgrid(np.arange(0, array_size * dx, dx), np.arange(0, array_size * dy, dy))
rx = rx.ravel()
ry = ry.ravel()
r = np.column_stack((rx, ry, np.zeros(N)))

# Matrix of array response vectors
A = np.exp(-1j * r @ k)

# Additive noise
sigma2 = 0.01  # Noise variance
n = np.sqrt(sigma2) * (np.random.randn(N, L) + 1j * np.random.randn(N, L)) / np.sqrt(2)


# Received signal
x = A @ m + n

# ========= (3) MUSIC ALGORITHM ========= %

# Sample covariance matrix
Rxx = x @ x.conj().T / L

# Eigendecompose
lambda_, E = np.linalg.eig(Rxx)
idx = lambda_.argsort()
lambda_ = lambda_[idx]
E = E[:, idx]
En = E[:, :-M]  # Noise eigenvectors (ASSUMPTION: M IS KNOWN)

# MUSIC search directions
AzSearch = np.arange(0, 181)  # Azimuth values to search
ElSearch = np.arange(0, 91)  # Elevation values to search

# 2D MUSIC spectrum
Z = np.zeros((len(AzSearch), len(ElSearch)))

for i, el_value in enumerate(ElSearch):
    # Elevation search value
    el = el_value

    # Points on azimuth array manifold curve to search (for this el)
    kSearch = np.pi * np.column_stack((np.cos(np.deg2rad(AzSearch)) * np.cos(np.deg2rad(el)),
                                       np.sin(np.deg2rad(AzSearch)) * np.cos(np.deg2rad(el)),
                                       np.sin(np.deg2rad(el)) * np.ones_like(AzSearch))).T
    ASearch = np.exp(-1j * r @ kSearch)

    # Compute azimuth spectrum for this elevation
    Z[:, i] = np.sum(np.abs(ASearch.conj().T @ En) ** 2, axis=1)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
AzSearch, ElSearch = np.meshgrid(AzSearch, ElSearch)
ax.plot_surface(AzSearch, ElSearch, -10 * np.log10(Z.T / N), cmap='viridis')
ax.set_title('2D MUSIC Example with 4x4 Antenna Array')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_zlabel('MUSIC spectrum (dB)')
plt.show()
