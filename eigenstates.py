from numpy import *
from numpy.linalg import eig, norm
from scipy.special import hermite, gamma
from matplotlib.pyplot import *


def Part1():
    # Teil 1: Unteraufgaben a), b), c), d)

    # Exakte Loesungen
    def psi_exact(n, x):
        C = 1.0 / sqrt(2**n * gamma(n+1)) / sqrt(sqrt(pi))
        return C * exp(-0.5*x**2) * polyval(hermite(n), x)

    def E_exact(n):
        return n + 0.5

    # Harmonisches Potential
    v = lambda x: 0.5*x**2


    # Unteraufgabe a)

    def build_hamiltonian(N):
        r"""
        N: Number of 'unknowns' or 'grid points'
        """
        H = zeros((N,N))

        ##############################################
        #                                            #
        # TODO: Bauen Sie den diskreten Hamiltonian. #
        #                                            #
        ##############################################

        return H


    # Unteraufgabe b)

    N = 2**arange(5, 11)

    EW = []
    EV = []

    for n in N:
        ew = ones(n)
        ev = eye(n)

        #####################################################
        #                                                   #
        # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
        #                                                   #
        #####################################################

        # Store
        EW.append(ew)
        EV.append(ev)


    # Unteraufgabe c)

    nn = 32
    n = arange(nn)

    figure()

    #####################################
    #                                   #
    # TODO: Plotten Sie die Eigenwerte. #
    #                                   #
    #####################################

    grid(True)
    xlim(n.min(), n.max())
    legend(loc='best')
    xlabel(r"$n$")
    ylabel(r"$E_n$")
    savefig("harmonic_eigenvalues.png")


    figure()

    ################################################
    #                                              #
    # TODO: Plotten Sie die Fehler der Eigenwerte. #
    #                                              #
    ################################################

    grid(True)
    xlim(n.min(), n.max())
    legend(loc='best')
    xlabel(r"$n$")
    ylabel(r"$E_n$")
    savefig("harmonic_eigenvalues_error.png")


    # Unteraufgabe d)

    figure()

    ##########################################
    #                                        #
    # TODO: Plotten Sie die Eigenfunktionen. #
    #                                        #
    ##########################################

    grid(True)
    xlim(-6, 6)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("harmonic_eigenfunctions.png")


    figure()

    #####################################################
    #                                                   #
    # TODO: Plotten Sie die Fehler der Eigenfunktionen. #
    #                                                   #
    #####################################################

    grid(True)
    xlim(-6, 6)
    ylim(1e-10, 1e-3)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("harmonic_eigenfunctions_error.png")




def arnoldi(A, v0, k):
    r"""Arnoldi algorithm to compute the Krylov approximation :math:`H` of a matrix :math:`A`.

    :param A: The matrix :math:`A` of shape :math:`N \times N` to approximate.
    :param v0: The initial vector :math:`v_0` of length :math:`N`.
    :param k: The number :math:`k` of Krylov steps performed.
    :return: A tuple :math:`(V, H)` where :math:`V` is the large matrix of shape
             :math:`N \times (k+1)` containing the orthogonal vectors and :math:`H` is the
             small matrix of shape :math:`k \times k` containing the Krylov approximation
             of :math:`A`.
    """
    r, c = A.shape
    V = zeros((r, k+1), dtype=complexfloating)
    H = zeros((k+1, k), dtype=complexfloating)

    ########################################################
    #                                                      #
    # TODO: Implementieren Sie hier das Arnoldi Verfahren. #
    #                                                      #
    ########################################################

    return V, H[:-1,:]




def Part2():
    # Teil 1: Unteraufgaben e), f)

    # Morse Potential
    d = 16
    a = 1
    v = lambda x: d*(exp(-2*a*x) - 2*exp(-a*x))


    # Unteraufgabe e)

    N = 256
    x, h = linspace(-2, 8, N, retstep=True)

    ##############################################
    #                                            #
    # TODO: Bauen Sie den diskreten Hamiltonian. #
    #                                            #
    ##############################################

    #####################################################
    #                                                   #
    # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
    #                                                   #
    #####################################################


    figure()
    plot(x, v(x), "k")

    ##########################################
    #                                        #
    # TODO: Plotten Sie die Eigenfunktionen. #
    #                                        #
    ##########################################

    grid(True)
    ylim(-1, 2)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("morse_eigenfunctions.png")


    # Unteraufgabe f)

    # Startvektor fuer Arnoldi
    v0 = ones(N)/sqrt(N)

    #####################################################
    #                                                   #
    # TODO: Berechnen Sie Eigenwerte und Eigenvektoren  #
    #       mit dem Arnoldi Verfahren.                  #
    #                                                   #
    #####################################################


    figure()
    plot(x, v(x), "k")

    ##########################################
    #                                        #
    # TODO: Plotten Sie die Eigenfunktionen. #
    #                                        #
    ##########################################

    grid(True)
    ylim(-1, 2)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("morse_eigenfunctions_arnoldi.png")




def Part3():
    # Teil 1: Unteraufgaben g), h), i)

    # Harmonischer Osillator
    v = lambda x, y: 0.5*(x**2 + y**2)

    # Henon-Heiles
    a = 2.0
    b = 0.4
    v = lambda x, y: 0.5*a*(x**2 + y**2) + b*(x**2*y - y**3/3.0)

    N = 32
    x, h = linspace(-3, 3, N, retstep=True)
    X,Y = meshgrid(x,x)
    x = X.reshape(-1)
    y = Y.reshape(-1)

    # Just K for later use
    K = 6


    # Unteraufgabe g)

    H = zeros((N*N, N*N))

    ##############################################
    #                                            #
    # TODO: Bauen Sie den diskreten Hamiltonian. #
    #                                            #
    ##############################################

    figure()
    matshow(H[:50,:50])
    colorbar()
    savefig("discrete_hamiltonian.png")


    # Unteraufgabe h)

    Psi = eye(N*N)

    #####################################################
    #                                                   #
    # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
    #                                                   #
    #####################################################


    fig = figure(figsize=(12,8))
    for k in xrange(K):
        # Plot first K eigenstates
        psik = Psi[:,k].reshape((N,N))

        fig.add_subplot(2, 3, k+1)
        ax = fig.gca()
        ax.set_aspect('equal')

        ax.contour(X, Y, v(X,Y), colors="gray", levels=linspace(0, 20, 15))
        ax.contourf(X, Y, abs(psik), levels=linspace(0, 0.15, 40))

        ax.grid(True)
    fig.savefig("henon_eigenfunctions.png")


    # Unteraufgabe i)

    ewk = ones(N)
    Psik = eye(N*N)

    # Startvektor fuer Arnoldi
    v0 = ones(N*N)/N

    #####################################################
    #                                                   #
    # TODO: Berechnen Sie Eigenwerte und Eigenvektoren  #
    #       mit dem Arnoldi Verfahren.                  #
    #                                                   #
    #####################################################


    fig = figure(figsize=(12,8))
    vc = 0
    for k in xrange(0, 2*K):
        # Skip some numerical artefacts
        if abs(ewk[k]) < 1.0:
            continue
        else:
            vc += 1
            if vc > K:
                break

        # Plot first K eigenstates
        psik = Psik[:,k].reshape((N,N))

        fig.add_subplot(2, 3, vc)
        ax = fig.gca()
        ax.set_aspect('equal')

        ax.contour(X, Y, v(X,Y), colors="gray", levels=linspace(0, 20, 15))
        ax.contourf(X, Y, abs(psik), levels=linspace(0, 0.15, 40))

        ax.grid(True)
    fig.savefig("henon_eigenfunctions_arnoldi.png")




if __name__ == "__main__":

    #################################################################################
    #                                                                               #
    # TODO: Kommentieren Sie geloeste / nicht geloeste Teile der Aufgabe ein / aus. #
    #       Dies kann beim Testen Rechenzeit sparen!                                #
    #                                                                               #
    #################################################################################

    Part1()
    #Part2()
    #Part3()
