class AddGate:
    def __init__(self):
        self.x = self.y = None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x + y

    def backward(self, d_out):
        return d_out, d_out


class MultiplyGate:
    def __init__(self):
        self.x = self.y = None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, d_out):
        return d_out * self.y, d_out * self.x


class PowerGate:
    def __init__(self, power):
        self.x = None
        self.power = power

    def forward(self, x):
        self.x = x
        return x ** self.power

    def backward(self, d_out):
        return d_out * self.power * (self.x ** (self.power - 1))


# Inputs
w, b, x, y = 2, 8, -2, 2

# Gates
mul1 = MultiplyGate()
add1 = AddGate()
add2 = AddGate()
pow_gate = PowerGate(2)
mul2 = MultiplyGate()

# Forward pass
c = mul1.forward(w, x)       # c = w * x
a = add1.forward(c, b)       # a = c + b
d = add2.forward(a, -y)      # d = a - y
e = pow_gate.forward(d)      # e = d^2
loss = mul2.forward(0.5, e)  # Loss = 0.5 * e
print(f"Loss: {loss}")

# Backward pass
_, de = mul2.backward(1)        # Grad wrt e
dd = pow_gate.backward(de)      # Grad wrt d
da, _ = add2.backward(dd)       # Grad wrt a
dc, db = add1.backward(da)      # Grad wrt c, b
dw, dx = mul1.backward(dc)      # Grad wrt w, x

# Display gradients
print(f"A={de}, B={dd},  C={da}, D={dc} , E={db}, F={dw}")