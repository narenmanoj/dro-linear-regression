import numpy as np
import typing

def _construct_row_map(matrices: np.array | typing.List[np.array]):
  row_counts = [matrix.shape[0] for matrix in matrices]
  cumulative_row_counts = np.cumsum([0] + row_counts)
  row_index_map = {
    i: list(range(cumulative_row_counts[i], cumulative_row_counts[i + 1]))
    for i in range(len(matrices))
  }
  return row_index_map

def _reverse_row_map(row_index_map):
  # TODO: implement less stupidly
  ans = []
  for k in row_index_map.keys():
    for vi in row_index_map[k]:
      ans.append(k)
  return np.array(ans)

def leverage_scores(mat: np.array):
  assert len(mat.shape) == 2, "Input must be a matrix. Received something with shape {mat.shape}"
  inv = np.linalg.inv(mat.T @ mat)
  return np.diag(mat @ inv @ mat.T)

def solve_system(mat: np.array, vec: np.array):
  return np.linalg.pinv(mat) @ vec

def block_lewis_weights(mat: np.array | typing.List[np.array], p):
  assert len(mat.shape) == 3, "Input must be a 3D numpy array. Received something with shape {mat.shape}"
  index_map = _construct_row_map(mat)
  reverse_index_map = _reverse_row_map(index_map)
  A = np.concatenate(mat, axis=0)
  m = mat.shape[0]
  n = A.shape[0]
  d = A.shape[1]
  print(f"m = {m}, n = {n}, d = {d}")
  T = int(np.ceil(3 * np.log(m))) # TODO: what's a constant that's good enough
  b = d / m * np.ones(n)
  b_vec = [b]
  for t in range(T):
    b_prev = b_vec[-1]
    lev_scores = leverage_scores(np.diag(np.power(b_prev, 0.5 - 1.0 / p)) @ A)
    lev_scores = lev_scores * d / sum(lev_scores)
    new_weights = np.array([sum(lev_scores[j] for j in index_map[i]) for i in range(m)])
    b_new = np.array([new_weights[reverse_index_map[i]] for i in range(len(reverse_index_map))])
    b_vec.append(b_new)
  return 1.1 * sum(b_vec) / T

class DROLinearRegression:
  def __init__(self, 
               design: typing.List[np.array], 
               response: typing.List[np.array],
               config: typing.Dict):
    # dimension validations
    self.dim = design[0].shape[1]
    for des in design:
      assert des.shape[1] == self.dim, f"Dimension mismatch in design. Expected {self.dim} but got the matrix {des}"
    assert len(design) == len(response), f"Received {len(design)} designs and {len(response)} responses"

    # set everything up
    self.design = np.array(design)
    self.stacked_design = np.concatenate(self.design, axis=0)
    self.stacked_response = np.concatenate(response)
    self.row_map = _construct_row_map(self.design)
    self.reverse_row_map = _reverse_row_map(self.row_map)
    self.response = np.array(response)
    self.config = config
    if "p" not in self.config:
      self.p = np.inf
    else:
      trial_p = self.config["p"]
      try:
        if trial_p != np.inf:
          trial_p = float(trial_p)
        assert trial_p >= 2.0 or trial_p == np.inf
        self.p =  trial_p
      except Exception as e:
        print(f"Error in deciding p. Config p must be a floatable between 2 and np.inf. Received: {trial_p}")
    if "GeometryType" in self.config:
      self.geometry_type = self.config["GeometryType"]
      assert self.geometry_type in ["Trivial", "Lewis"], f"invalid geometry type passed {self.geometry_type}"
    else:
      self.geometry_type = "Trivial"
    self.num_problems = len(design)
    self.num_rows = self.stacked_design.shape[0]

    self.x = np.zeros(self.dim)
    self.compute_geometry()
    self.warm_start()

  def objective(self, input_point: np.array, powered: bool=False):
    l2norms = np.linalg.norm(self.design @ input_point - response, axis=1, ord=2)
    unpowered = np.linalg.norm(l2norms, ord=self.p)
    if powered:
      return np.power(unpowered, self.p)
    return unpowered

  def compute_geometry(self):
    if self.geometry_type == "Trivial":
      self.w = np.ones(self.num_rows)
    elif self.geometry_type == "Lewis":
      # do one other thing
      appended_response = np.concatenate([self.design, self.response[:, :, np.newaxis]], axis=2)
      self.w = block_lewis_weights(appended_response, self.p)

  def warm_start(self):
    raised = np.power(self.w, 0.5 - 1.0 / self.p)
    W_powered_left = np.diag(raised)
    self.x = solve_system(W_powered_left @ self.stacked_design, W_powered_left @ self.stacked_response)

  def step(self) -> np.array:
    # perform a single iteration of the algorithm and update the internal state variables
    # TODO: implement
    return np.array([0.0])