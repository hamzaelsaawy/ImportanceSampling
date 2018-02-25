#
# Helper functions
#

all_equal(f, xs) = all(x -> f(x) == f(first(xs)), xs[2:end])
all_equal(xs) = all_equal(identity, xs)
