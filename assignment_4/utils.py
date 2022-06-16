class AsMatrix():
    def __init__(
        self,
        operator,
        adjoint,
    ):
        self.operator = operator
        self.adjoint = adjoint

    def __matmul__(
        self,
        x,
    ):
        return self.operator(x)

    @property
    def T(
        self,
    ):
        return AsMatrix(self.adjoint, self.operator)
