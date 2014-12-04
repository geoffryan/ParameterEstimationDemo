import numpy as np
import matplotlib.pyplot as plt
import emcee as em
import triangle as tri

def line(x, par):
    return par[0] + par[1]*x

def logprob(par, f, x, y, yerr):
    diff = (f(x, par) - y) / yerr
    return -0.5 * (diff*diff).sum()

nwalkers = 100
ndim = 2
run = 250

b = 1.0
m = 2.0
sig = 0.1
pars = [b,m]

x = np.linspace(0,1,10)
y = line(x, [b,m]) + sig * np.random.randn(x.shape[0])
yerr = sig*np.ones(x.shape)


p0 = 0.01*np.random.randn(nwalkers, ndim)

print("Sampling...")
sampler = em.EnsembleSampler(nwalkers, ndim, logprob, args=[line, x, y, yerr])
sampler.run_mcmc(p0, run)

samples = sampler.chain[:,100:,:].reshape(-1,ndim)

N = samples.shape[0]
M = samples[:,1].sum() / N
B = samples[:,0].sum() / N
M2 = (samples[:,1]*samples[:,1]).sum() / N
B2 = (samples[:,0]*samples[:,0]).sum() / N

print("\nm = {0:.3f}".format(pars[1]))
print("m = {0:.3f} +/- {1:.3f}\n".format(M, np.sqrt(M2-M*M)))
print("b = {0:.3f}".format(pars[0]))
print("b = {0:.3f} +/- {1:.3f}\n".format(B, np.sqrt(B2-B*B)))

ytrue = line(x, pars)

print("Plotting...")
plt.figure()
plt.errorbar(x, y, yerr, fmt='k.')
plt.plot(x, ytrue, 'r')
plt.savefig("true.png")
plt.close()

tri.corner(sampler.flatchain, labels=[r'$b$', r'$m$'], truths=[b,m])
plt.savefig("corner_all.png")
plt.close()

tri.corner(sampler.chain[:,100:,:].reshape(-1,ndim), labels=[r'$b$', r'$m$'], truths=[b,m])
plt.savefig("corner_converged.png")
plt.close()

nstep = 10
for i in xrange(0,run,nstep):
    plt.figure()
    plt.errorbar(x, y, yerr, fmt='b.')
    for j in xrange(nwalkers):
        plt.plot(x, line(x,sampler.chain[j,i]), 'k-', alpha=0.1)
    plt.savefig("steps{0:03d}.png".format(i))
plt.close()
print("Done.")
