import executorch.exir as exir
from executorch.exir.tests.models import Mul
m = Mul()
print(exir.capture(m, m.get_random_inputs()).to_edge())
open("mul.pte", "wb").write(exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer)