from numpy import asarray, histogram, digitize
from random import gauss, randint
import MI as mi


def test():
    x = asarray([gauss(0, 1) for i in range(1000)])
    y1 = asarray([int(e > 0) for e in x])
    y2 = asarray([randint(0, 1) for e in x])

    hx, bx = histogram(x, bins=x.size / 10, density=True)
    dx = digitize(x, bx)

    print "X ~ N(0,1)"
    print "y1 = 1 <=> x > 0"
    print "y2 = 1 con probabilidad 0.5"
    print
    print "I(y1;x) = H(X) - H(X|Y1) = %.02f" % (mi.mutual_information(x, y1))
    print "I(y1;x) = H(Y1) - H(Y1|X) = %.02f" % (mi.mutual_information(y1, dx))
    print
    print "I(y2;x) = H(X) - H(X|Y2) = %.02f" % (mi.mutual_information(x, y2))
    print "I(y2;x) = H(Y2) - H(Y2|X) = %.02f" % (mi.mutual_information(y2, dx))

test()