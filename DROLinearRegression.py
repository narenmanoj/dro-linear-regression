import numpy as np
import typing

def _construct_row_map(matrices: np.array):
  row_counts = [matrix.shape[0] for matrix in matrices]
  cumulative_row_counts = np.cumsum([0] + row_counts)
  row_index_map = {
    i: range(cumulative_row_counts[i], cumulative_row_counts[i+1])
    for i in range(len(matrices))
  }
  return row_index_map

def leverage_scores(mat: np.array):
  assert len(mat.shape) == 2, "Input must be a matrix. Received something with shape {mat.shape}"
  inv = np.linalg.inv(mat.T @ mat)
  return np.diag(mat @ inv @ mat.T)

def block_lewis_weights(mat: np.array, p):
  assert len(mat.shape) == 3, "Input must be a 3D numpy array. Received something with shape {mat.shape}"
  index_map = _construct_row_map(mat)
  A = np.concatenate(mat, axis=0)
  m = mat.shape[0]
  n = A.shape[0]
  d = A.shape[1]
  T = 3 * np.ln(m) # TODO: what's a constant that's good enough
  b = d / m * np.ones(n)
  b_vec = [b]
  for t in range(T):
    b_prev = b_vec[-1]
    lev_scores = leverage_scores(np.diag(np.power(b_prev, 0.5 - 1.0 / p)) * A)

    b_new = 0.0 # TODO: aggregate

    b_vec.append(b_new)
  return 1.1 * sum(b_vec) / T

class DROLinearRegression:
  def __init__(self, 
               design: typing.List[np.array], 
               response: typing.List[np.array],
               config: typing.Dict):
    self.design = np.array(design)
    self.response = np.array(response)
    self.config = config
    if "p" not in self.config:
      self.p = np.inf
    else:
      trial_p = self.config["p"]
      try:
        self.p = np.float(self.config["p"])
        assert self.p >= 2.0
      except Exception as e:
        print(f"Error in deciding p. Config p must be a floatable between 2 and np.inf. Received: {trial_p}")
    if "GeometryType" in self.config:
      self.geometry_type = self.config["GeometryType"]
      assert self.geometry_type in ["Trivial", "Lewis"], f"invalid geometry type passed {self.geometry_type}"
    else:
      self.geometry_type = "Trivial"

    # dimension validations
    self.dim = design[0].shape[1]
    for des in design:
      assert des.shape[1] == self.dim, f"Dimension mismatch in design. Expected {self.dim} but got the matrix {des}"
    assert len(design) == len(response), f"Received {len(design)} designs and {len(response)} responses"
    self.num_problems = len(design)

    self.x = np.zeros(self.dim)

  def objective(self, input_point: np.array, powered: bool=False):
    l2norms = np.linalg.norm(self.design @ input_point - response, axis=1, ord=2)
    unpowered = np.linalg.norm(l2norms, ord=self.p)
    if powered:
      return np.power(unpowered, self.p)
    return unpowered

  def compute_geometry(self):
    # TODO: implement
    if self.geometry_type == "Trivial":
      # do one thing
      pass
    elif self.geometry_type == "Lewis":
      # do one other thing
      pass
    return None

  def initialize(self):
    # TODO: implement
    return None

  def step(self) -> np.array:
    # perform a single iteration of the algorithm and update the internal state variables
    # TODO: implement
    return np.array([0.0])