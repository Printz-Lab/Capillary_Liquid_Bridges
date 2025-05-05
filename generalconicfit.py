from numpy.linalg import svd, qr, solve, det
from numpy import array, matmul, float64, transpose # , allclose, matmul, float64
from math import sqrt



SMALL = 10**-12
sqrt2 = 2**0.5
TYPE_HYPERBOLA = "hyperbola"
TYPE_PARABOLA = "parabola"
TYPE_ELLIPSE = "ellipse"
TYPE_LINEAR = "linear"
def centered(points) :
    """Subtract the average from the (x,y) pairs."""
    xs_uncentered = [x for (x,y) in points]
    ys_uncentered = [y for (x,y) in points]

    x_center = sum([x for x in xs_uncentered])/len(points)
    y_center = sum([y for y in ys_uncentered])/len(points)

    xs = [x-x_center for x in xs_uncentered]
    ys = [y-y_center for y in ys_uncentered]

    return list(zip(xs,ys)), (x_center, y_center)

def translate_conic(conic, displacement) :
    """Return a conic that has been translated by an amount (dx,dy)."""
    (dx, dy) = displacement
    A,B,C,D,E,F = conic.get("coeffs")
    new_coeffs = (A,B,C,
                  D - 2*A*dx - B*dy,
                  E - 2*C*dy - B*dx,
                  A*dx*dx + B*dx*dy + C*dy*dy - D*dx - E*dy + F)
    conic["coeffs"] = new_coeffs
    conic["center"] = np.array(conic.get("center",(0,0)))+ np.array(displacement)
    return conic

def collinearity(points) :

    """Return a number indicating how very nearly the points follow a
line. Zero indicates perfect fit. Also return the best-fit (Theil-Sen
estimator) line in point-slope form."""

    (x0, y0) = points[0]

    average_x_deviation = sum(abs(x-x0) for (x,y) in points[1:])/len(points)
    if (average_x_deviation < SMALL) :
        # VERTICAL LINE
        return 0, (0,1), points[0]

    slopes = sorted([(y-y0)/float(x-x0) for (x,y) in points[1:]])
    median_slope = slopes[int(len(slopes)/2)]

    intercepts = sorted([y - median_slope*x for (x,y) in points[1:]])
    median_intercept = intercepts[int(len(intercepts)/2)]

    print(">>>", median_slope, median_intercept)
    average_deviation = sum((y-median_slope*x-median_intercept)**2/len(points)
                            for (x,y) in points)

    return average_deviation, (1, median_slope), (0, median_intercept)

def smallest_singular(M, compute_singular_vector=True) :
    """Return a pair containing the smallest singular value of M and a
corresponding singular vector.
    """

    if compute_singular_vector: 
        _, s, Vh = svd(M, full_matrices=True)
        return s[-1], Vh[-1]

def fit_conic(points, desired_type=None) :
    points, center = centered(points)

    matrix_monomials = array([
        [1, x, y,    y*y-x*x, 2*x*y, y*y+x*x]
        for (x,y) in points])

    # ---- Check for degenerate conic lying on a single line
    fit, slope, intercept = collinearity(points)
    print("COLLINEARITY", fit, slope, intercept)
    if (fit < SMALL) :
        return translate_conic({"type" : TYPE_LINEAR,
                                "coeffs" : (0, 0, 0, slope[1], -slope[0], slope[0]*intercept[1] - slope[1]*intercept[0]),
                                "fit" : fit,        
                                }, center)

        return None


    # ---- Perform decomposition
    submatrix = matrix_monomials[:,:3] # (1,x,y)
    q,r = qr(submatrix, "complete")


    R_full = matmul(transpose(q), matrix_monomials)
    RA = R_full[:3, 0:3]
    RB = R_full[:3, 3:6]
    RC = R_full[3:, 3:6]
    _, vec_smallest = smallest_singular(RC)

    # ----- Determine type of conic
    Z = array([[-1,0,1],
               [0,sqrt2,0],
               [1,0,1]]) / sqrt2
    D = array([[-1,0,0],
               [0,-1,0],
               [0,0,1]])/4

    A_det = transpose(vec_smallest) @ D @ vec_smallest
    A_trace = vec_smallest[2]

    conic_type = TYPE_PARABOLA if (abs(A_det) < SMALL) else TYPE_ELLIPSE if A_det > 0 else TYPE_HYPERBOLA


    # ---- Adjust answer if it doesn't match desired conic type (not
    # ---- implemented yet)

    if(False and desired_type == TYPE_PARABOLA) :
        # "desired type" not implemented yet.  also not necessary for
        # this application b/c empirical points are expected to be
        # extremely unambiguously close to particular conics.

        pass
    else :
        q = vec_smallest


    # ------ Solve for conic coefficients
    ret = solve(RA, -1 * (RB @ q)) # c, *two_b

    params = [*ret, *q]

    conic_coeffs = [ params[5]-params[3],
                         2*params[4],
                         params[5]+params[3],
                         params[1],
                         params[2],
                         params[0]]


    fit  = sum([x**2 for x in matmul(matrix_monomials, array(params))])/len(points)


    A,B,C,D,E,F = conic_coeffs


    quadratic_form = array([[A, B/2],[B/2,C]])

    full_form = array([[  A, B/2, D/2],
                       [B/2,   C, E/2],
                       [D/2, E/2,   F]])

    if(abs(det(full_form)) < 1e-8) :
            # degenerate conic
            return None


    u,s,vh = svd(quadratic_form)

    K =  -det(full_form)/det(quadratic_form) if det(quadratic_form) != 0 else None

    axes = None
    try :
        axes = [(K/x)**0.5 for x in s.tolist()]
    except Exception :
        pass

    return translate_conic({"type" : conic_type,
                            "coeffs" : conic_coeffs,
                            "fit" : fit,
                            "axes" : axes,
                            "diagonalizer" : vh.tolist()       
                            }, center)

def newton_root(f, init_guess=60) :
    x = init_guess
    small = SMALL
    for _ in range(30) :
        if abs(f(x)) < 1e-8 :
            break
        else :
            x -= f(x)*small/(f(x+small)-f(x))
    return x

def plot_fitted_conic(points) :
    conic = fit_conic(points)
    A,B,C,D,E,F = conic.get("coeffs")
    xs = np.arange(min([x for (x,y) in points]),
                   max([x for (x,y) in points]),
                   1).tolist()
    ys = [newton_root(lambda y: A*x*x + B*x*y + C*y*y + D*x + E*y + F) for x in xs]

    plt.scatter([x for (x,y) in points],
                [y for (x,y) in points])

    ys_smooth = make_interp_spline(xs, ys, k=3)(xs)
    plt.plot(xs, ys_smooth)

    plt.show()

