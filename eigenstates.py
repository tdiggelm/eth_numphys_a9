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

    def build_hamiltonian(N, h):
        r"""
        N: Number of 'unknowns' or 'grid points'
        """
        #H = zeros((N,N))

        ##############################################
        #                                            #
        # TODO: Bauen Sie den diskreten Hamiltonian. #
        #                                            #
        ##############################################
        L = diag(N*[-2])+eye(N,k=1)+eye(N,k=-1)
        L[0]=L[1]
        L[-1]=L[-2]
        L = 1/h**2*L
        H = -0.5*L+diag(v(x))

        return H


    # Unteraufgabe b)

    NN = 2**arange(5, 11)
    #NN = 2**arange(5, 7)

    EW = []
    EV = []

    a, b = -10.0, 10.0

    for N in NN:
        #ew = ones(N)
        #ev = eye(N)

        #####################################################
        #                                                   #
        # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
        #                                                   #
        #####################################################
        
        x, h = linspace(a, b, N, retstep=True)
        H = build_hamiltonian(N, h)
        
        ew, ev = eig(H)
        ew = abs(ew)
        ev = ev.T.real
        
        ew, ev = zip(*[(ew[i], ev[i]) for i in argsort(ew)])
        
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
    for N, E in zip(NN, EW):
        plot(E[:nn], label=r"$N=%d$" % N)

    title("harmonic eigenvalues")
    grid(True)
    xlim(n.min(), n.max())
    legend(loc='best')
    xlabel(r"$n$")
    ylabel(r"$E_n^{(N)}$")
    savefig("harmonic_eigenvalues.pdf")


    figure()

    ################################################
    #                                              #
    # TODO: Plotten Sie die Fehler der Eigenwerte. #
    #                                              #
    ################################################
    Ee = E_exact(arange(nn))
    for N, E in zip(NN, EW):
        plot(abs(E[:32]-Ee), label=r"$N=%d$" % N)

    title("harmonic eigenvalues error")
    grid(True)
    xlim(n.min(), n.max())
    legend(loc='best')
    xlabel(r"$n$")
    ylabel(r"$|E_n^{(N)}-E_n^{exact}|$")
    savefig("harmonic_eigenvalues_error.pdf")


    # Unteraufgabe d)

    nn = 7
    x = linspace(a, b, 1024)
    N, E, psi = NN[-1], EW[-1], EV[-1]
    psi_e = array([psi_exact(i,x) for i in range(nn)])
    psi_e = (psi_e.T / norm(psi_e, axis=1)).T


    figure()

    ##########################################
    #                                        #
    # TODO: Plotten Sie die Eigenfunktionen. #
    #                                        #
    ##########################################
    for n in range(nn):
        plot(x, psi[n], label="$n=%d$" % n)
    
    # plot levels
    #for n in range(nn):
    #    plot(x, E[n]+psi[n], label="$n=%d$" % n)
    #plot(x, v(x), "k")
    #ylim(0, 8)
    
    title("harmonic eigenfunctions")
    grid(True)
    xlim(-6, 6)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("harmonic_eigenfunctions.pdf")


    figure()

    #####################################################
    #                                                   #
    # TODO: Plotten Sie die Fehler der Eigenfunktionen. #
    #                                                   #
    #####################################################

    # log-plot? => do both
    for n in range(nn):
        semilogy(x, abs(abs(psi_e[n])-abs(psi[n])), label="$n=%d$" % n)

    title("harmonic eigenfunctions error")
    grid(True)
    xlim(-6, 6)
    ylim(1e-10, 1e-3)
    legend(loc='best')
    xlabel(r"$x$")
    ylabel(r"$\psi_n(x)$")
    savefig("harmonic_eigenfunctions_error.pdf")




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
    V[:,0] = v0/norm(v0) # V[:,0] = v0.copy/norm(v0)
    for m in xrange(k):
        vt = A.dot(V[:,m])
        for j in xrange(m+1):
            H[j,m] = dot(V[:,j].conjugate(), vt)
            vt -= H[j,m] * V[:,j]
        H[m+1,m] = norm(vt)
        V[:,m+1] = vt/H[m+1,m]
    return V[:,:-1], H[:-1,:]


def Part2():
    # Teil 1: Unteraufgaben e), f)

    # Morse Potential
    d = 16
    a = 1
    v = lambda x: d*(exp(-2*a*x) - 2*exp(-a*x))

    # Unteraufgabe e)

    N = 256
    x, h = linspace(-2, 8, N, retstep=True)

    for symmetric in [True, False]:
        how = "symmetric" if symmetric else "nonsymmetric"
        ##############################################
        #                                            #
        # TODO: Bauen Sie den diskreten Hamiltonian. #
        #                                            #
        ##############################################
        L = diag(N*[-2])+eye(N,k=1)+eye(N,k=-1)
        if not symmetric:
            L[0]=L[1]
            L[-1]=L[-2]
        L = 1/h**2*L
        H = -0.5*L+diag(v(x))
    
        #####################################################
        #                                                   #
        # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
        #                                                   #
        #####################################################
        ew, ev = eig(H)
        ew = abs(ew)
        ev = ev.T.real
        ew, ev = zip(*[(ew[i], ev[i]) for i in argsort(ew)])


        figure()
        plot(x, v(x), "k")

        ##########################################
        #                                        #
        # TODO: Plotten Sie die Eigenfunktionen. #
        #                                        #
        ##########################################
    
        for n in range(4):
            plot(x, ev[n], label="$n=%d$" % n)

        title("morse eigenfunctions %s" % how)
        grid(True)
        ylim(-1, 2)
        legend(loc='best')
        xlabel(r"$x$")
        ylabel(r"$\psi_n(x)$")
        savefig("morse_eigenfunctions_%s.pdf" % how)


        # Unteraufgabe f)

        # Startvektor fuer Arnoldi
        v0 = ones(N)/sqrt(N)

        #####################################################
        #                                                   #
        # TODO: Berechnen Sie Eigenwerte und Eigenvektoren  #
        #       mit dem Arnoldi Verfahren.                  #
        #                                                   #
        #####################################################

        V, HH = arnoldi(H, v0, 150)
        ew, ev = eig(HH)
        ew = abs(ew)
        ev = V.dot(ev) # transform ev's back to initial base
        ev = ev.T.real
        ew, ev = zip(*[(ew[i], ev[i]) for i in argsort(ew)])

        figure()
        plot(x, v(x), "k")

        ##########################################
        #                                        #
        # TODO: Plotten Sie die Eigenfunktionen. #
        #                                        #
        ##########################################
        for n in range(4):
            plot(x, ev[n], label="$n=%d$" % n)

            title("morse eigenfunctions arnoldi %s" % how)
        grid(True)
        ylim(-1, 2)
        legend(loc='best')
        xlabel(r"$x$")
        ylabel(r"$\psi_n(x)$")
        savefig("morse_eigenfunctions_arnoldi_%s.pdf" % how)




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
    V = diag(array([v(x[i],x[j]) for j in range(N) for i in range(N)]))
    L=-4*eye(N*N)+eye(N*N,k=1)+eye(N*N,k=-1) +eye(N*N,k=3)+eye(N*N,k=-3)
    L = 1/h**2*L
    H = -0.5*L+V


    figure()
    matshow(H[:50,:50])
    colorbar()
    savefig("discrete_hamiltonian.pdf")


    # Unteraufgabe h)

    Psi = eye(N*N)

    #####################################################
    #                                                   #
    # TODO: Berechnen Sie Eigenwerte und Eigenvektoren. #
    #                                                   #
    #####################################################
    #from scipy.sparse.linalg import eigs # use eigs for better performance
    #ew, ev = eigs(H, which="SR") # find eigenvalues with smallest real parts
    ew, ev = eig(H)
    ew = abs(ew)
    ev = ev.T.real    
    ew, ev = zip(*[(ew[i], ev[i]) for i in argsort(ew)])
    Psi = array(ev).T.real


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
    fig.savefig("henon_eigenfunctions.pdf")


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
    V, HH = arnoldi(H, v0, 180)
    ewk, Psik = eig(HH)
    ewk = abs(ewk)
    Psik = V.dot(Psik) # transform ev's back to initial base
    Psik = Psik.T.real
    ewk, Psik = zip(*[(ewk[i], Psik[i]) for i in argsort(ewk)])
    Psik = array(Psik).T.real

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
    fig.savefig("henon_eigenfunctions_arnoldi.pdf")




if __name__ == "__main__":

    #################################################################################
    #                                                                               #
    # TODO: Kommentieren Sie geloeste / nicht geloeste Teile der Aufgabe ein / aus. #
    #       Dies kann beim Testen Rechenzeit sparen!                                #
    #                                                                               #
    #################################################################################

    Part1()
    Part2()
    Part3()
