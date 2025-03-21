import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# i
mean_1 = 5
num_realizations = 1000
poisson_rv = sp.stats.poisson(mu=mean_1)
random_realizations = poisson_rv.rvs(size=num_realizations)

# PMF
x = np.arange(0, 15, 1)  # Range of values to evaluate PMF
pmf = poisson_rv.pmf(x)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(x, pmf)
ax1.set_title("PMF")
ax1.set_xlabel("x")
ax1.set_ylabel("PMF(x)")

# CDF
cdf = poisson_rv.cdf(x)

ax2.step(x, cdf, where="post")
ax2.set_title("CDF")
ax2.set_xlabel("x")
ax2.set_ylabel("CDF(x)")

# Histogram
ax3.hist(random_realizations, bins=np.arange(-0.5, 15.5, 1), density=True, edgecolor="black")
ax3.set_title("Histogram")
ax3.set_xlabel("x")
ax3.set_ylabel("Frequency")

plt.suptitle("Poisson Distribution")
plt.show()

# j
mean_2 = 7
std_dev = 2
normal_rv = sp.stats.norm(loc=mean_2, scale=std_dev)
normal_random_realizations = normal_rv.rvs(size=num_realizations)

# PDF
x = np.arange(0, 15, 1)  # Range of values to evaluate PMF
pdf = normal_rv.pdf(x)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(x, pdf)
ax1.set_title("PDF")
ax1.set_xlabel("x")
ax1.set_ylabel("PDF(x)")

# CDF
norm_cdf = normal_rv.cdf(x)

ax2.step(x, norm_cdf, where="post")
ax2.set_title("CDF")
ax2.set_xlabel("x")
ax2.set_ylabel("CDF(x)")

# Histogram
ax3.hist(normal_random_realizations, bins=np.arange(-0.5, 15.5, 1), density=True, edgecolor="black")
ax3.set_title("Histogram")
ax3.set_xlabel("x")
ax3.set_ylabel("Frequency")

plt.suptitle("Normal Distribution")
plt.show()

# k
np.random.seed(42)
dataset1 = np.random.normal(loc=0, scale=1, size=100)  # Mean = 0, Std Dev = 1
dataset2 = np.random.normal(loc=0.5, scale=1, size=100)  # Mean = 0.5, Std Dev = 1

# Perform the independent t-test
t_statistic, p_value = sp.stats.ttest_ind(dataset1, dataset2)

# Output the results
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05    # cutoff value
if p_value < alpha:
    print("Reject")
else:
    print("Accept")
