import os

import h5py
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(args, surface_path):
    result_file_path = os.path.join(surface_path, "2D_images/")
    if not os.path.isdir(result_file_path):
        os.makedirs(result_file_path)
    surf_name = args.surf_name

    with h5py.File(os.path.join(surface_path, "3d_surface_file.h5"), "r") as f:

        # Z_LIMIT = 10

        x = np.array(f["xcoordinates"][:])
        y = np.array(f["ycoordinates"][:])

        X, Y = np.meshgrid(x, y)

        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        else:
            print("%s is not found in %s" % (surf_name, surface_path))

        Z = np.array(f[surf_name][:])
        # Z[Z > Z_LIMIT] = Z_LIMIT
        # if "ssim" in surf_name:
        # Z = - np.log(1 - Z)  # logscale
        # elif "psnr" in surf_name:
        # Z = 50 - Z
        # Z = (Z - Z.min()) * 10 / (Z.max() - Z.min())

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(
            X, Y, Z, cmap="summer", levels=np.arange(args.vmin, args.vmax, args.vlevel)
        )
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(
            result_file_path + surf_name + "_2dcontour" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

        fig = plt.figure()
        CS = plt.contourf(X, Y, Z, levels=np.arange(args.vmin, args.vmax, args.vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(
            result_file_path + surf_name + "_2dcontourf" + ".pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

        # Save 2D heatmaps image
        plt.figure()
        sns_plot = sns.heatmap(
            Z,
            cmap="viridis",
            cbar=True,
            vmin=args.vmin,
            vmax=args.vmax,
            xticklabels=False,
            yticklabels=False,
        )
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(
            result_file_path + surf_name + "_2dheat.pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        fig.savefig(
            result_file_path + surf_name + "_3dsurface.pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )
