# Units
kN = 1.0
m = 1.0
sec = 1.0
mm = 0.001 * m
cm = 0.01 * m
g = 9.81 * m / sec ** 2
ton = kN * sec ** 2 / m
kg = 0.001 * kN * sec ** 2 / m
kPa = 1.0 * (kN / m ** 2)
MPa = 1.0e3 * (kN / m ** 2)
GPa = 1.0e6 * (kN / m ** 2)

# Material Properties
E = 25000 * MPa
v = 0.2
G = 0.5 * E / (1 + v)
