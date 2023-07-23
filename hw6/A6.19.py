import cvxpy as cp

# read in image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img = mpimg.imread("flower.png")
img = img[:, :, 0:3]
m, n, _ = img.shape

np.random.seed(5)
known_ind = np.where(np.random.rand(m, n) >= 0.90)
# grayscale image
M = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
# known color values
R_known = img[:, :, 0]
G_known = img[:, :, 1]
B_known = img[:, :, 2]
R_known = R_known[known_ind]
G_known = G_known[known_ind]
B_known = B_known[known_ind]


def save_img(filename, R, G, B):
    img = np.stack((np.array(R), np.array(G), np.array(B)), axis=2)
    # turn off ticks and labels of the figure
    plt.tick_params(
        axis="both", which="both", labelleft="off", labelbottom="off", bottom="off", top="off", right="off", left="off"
    )
    fig = plt.imshow(img)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)


R_given = np.copy(M)
R_given[known_ind] = R_known
G_given = np.copy(M)
G_given[known_ind] = G_known
B_given = np.copy(M)
B_given[known_ind] = B_known
save_img("flower_given.png", R_given, G_given, B_given)


def main():
    R = cp.Variable(shape=(m, n))
    G = cp.Variable(shape=(m, n))
    B = cp.Variable(shape=(m, n))

    constraints = [
        0.299 * R + 0.587 * G + 0.114 * B == M,
        R[known_ind] == R_known,
        G[known_ind] == G_known,
        B[known_ind] == B_known,
        0 <= R,
        R <= 1,
        0 <= G,
        G <= 1,
        0 <= B,
        B <= 1,
    ]

    problem = cp.Problem(cp.Minimize(cp.tv(R, G, B)), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(f"Objective value: {problem.value}")
    save_img("flower_reconstructed.png", R.value, G.value, B.value)


if __name__ == "__main__":
    main()
