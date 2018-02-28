#
# helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)

outer(v, w=v) = A_mul_Bt(v, w)
outer!(A, v, w=v) = A_mul_Bt!(A, v, w)
