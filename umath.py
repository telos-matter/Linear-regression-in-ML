from typing import Callable as function
import statistics
from sklearn.datasets import make_regression as sklearn_make_regression

def make_regression (n_samples: int= 100) -> tuple[list[float], list[float]]:
      x, y = sklearn_make_regression(n_samples= n_samples, n_features= 1, noise= 5)
      
      return x.flatten().tolist(), y.tolist()

def get_model_params (model) -> tuple[float, float]:
      b = model.predict([[0]])
      a = model.predict([[1]]) -b

      return (float(a[0]), float(b[0]))

def get_optimal_params (x: list[float], y: list[float]) -> tuple[float, float]:
      a = (statistics.covariance(x, y)*(len(x) -1)/(len(x)))/statistics.pvariance(x)
      b = statistics.mean(y) -a*statistics.mean(x)

      return (a, b)

def compute_line (x: list[float], a: float, b: float) -> list[float]:
      return [a*_x +b for _x in x if True]

# ---------------------------------------------------

def next_newton (f: function, df: function, xn: float) -> float:
      return xn -(f(xn)/df(xn))

def newton_fit (x: list[float], y: list[float], max_iter: int= 100, verbose: bool= True) -> tuple[float, float]:
      # Calculate constants
      sum_x_squared = sum([_x**2 for _x in x if True])
      sum_x = sum(x)
      sum_y = sum(y)
      sum_x_sum_y = 0
      for _x, _y in zip(x, y):
            sum_x_sum_y += _x * _y
      m = len(x)

      # Constant modified second partial derivative relative to a
      def dda (a: float) -> float:
            return sum_x_squared

      # Constant second partial derivative relative to b
      def ddb (b: float) -> float:
            return 1
      
      # ~Fitting~
      a = b = previous_a = previous_b = 0
      for _ in range(max_iter):

            # Modified first partial derivative relative to a
            da = lambda _a: _a*sum_x_squared +b*sum_x -sum_x_sum_y

            a = next_newton(da, dda, a)
            
            # Modified first partial derivative relative to b
            db = lambda _b: (a*sum_x +_b*m -sum_y)/m

            b = next_newton(db, ddb, b)

            if previous_a == a and previous_b == b:
                  if verbose:
                        print(f'Converged in {_} iterations.')
                  break

            previous_a = a
            previous_b = b
      else:
            if verbose:
                  print(f'Iterated for {max_iter} iterations and there is still room for convergence.')

      return a, b      
