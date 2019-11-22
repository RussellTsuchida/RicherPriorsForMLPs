import numpy as np

class DynamicGrid(object):
    """
    A 2D grid that aims to roughly track the shape of a loss surface for 
    plotting. When plotting the marginal likelihood/posterior of the GP,
    the surface becomes "smaller" and "more ill-conditioned" with increasing
    depth. Therefore, at every iteration, we set the top-left and bottom-right
    corners of the grid such that they are some percentage of the maximum.

    Follow the convention of [x, y], i.e. [horizontal, vertical].
    """
    def __init__(self, top_left, bottom_right, grid_size = 20, 
            p = 0.2):
        """
        Args:
            top_left (nparray): 2d array of floats, top left corner of grid.
            bottom_right (nparray): 2d array of floats, bottom right corner of 
                grid.
            grid_size (int): number of discrete points in each direction.
            e.g. 20 means 400 discrete grid points.
            p (float): percentage of maximum that the grid border points should
                sit at.
        """
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.p = p
        self.grid_size = grid_size

    def one_dimensional_grids(self):
        """
        Returns:
            list [grid1, grid2] of 1d nparrays representing grids of each 
                dimension.
        """
        grid_horz = np.linspace(self.top_left[0], self.bottom_right[0], 
                self.grid_size)
        grid_vert = np.linspace(self.bottom_right[1], self.top_left[1],
                self.grid_size)
        # Always include 0 mean point
        grid_vert = np.hstack((grid_vert, 0))
        grid_vert = np.unique(grid_vert)
        grid_vert.sort()

        return [grid_horz, grid_vert]

    def two_dimensional_grid(self):
        """
        Returns:
            two nparrays each of size (grid_size, grid_size) representing a 
                meshgrid over the space.
        """
        grid_horz, grid_vert = self.one_dimensional_grids()
        mesh_horz, mesh_vert = np.meshgrid(grid_horz, grid_vert)

        return [mesh_horz, mesh_vert]

    def update_corners(self, likelihood, log, grid_size = 100):
        """
        Find the grid points where the likelihood is equal to self.p and set
        these points to be the new corners.

        Args:
            likelihood (nparray): of size (self.grid_size, self.grid_size)
            log (bool): True for log likelihood, False for likelihood.
            grid_size (int): number of discrete points in each direction.
        """
        if log:
            likelihood = np.exp(likelihood)
        likelihood = likelihood - np.amin(likelihood) # make nonnegative
        amax = np.amax(likelihood)
        interesting_points = np.where(likelihood >= amax*self.p)

        top     = np.amax(interesting_points[0])
        bot     = np.amin(interesting_points[0])
        left    = np.amax(interesting_points[1])
        right   = np.amin(interesting_points[1])

        grid_horz, grid_vert = self.one_dimensional_grids()

        # Max and Min to ensure we always include mu = 0 visualisation
        self.top_left = [grid_horz[left], max(grid_vert[top], 0.1)]
        self.bottom_right = [grid_horz[right], min(grid_vert[bot], -0.1)]
        
        self.grid_size = grid_size


