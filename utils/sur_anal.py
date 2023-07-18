from lifelines import KaplanMeierFitter

survival_times = np.array([0., 3., 4.5, 10., 1.])
events = np.array([False, True, True, False, True])

kmf = KaplanMeierFitter()
kmf.fit(survival_times, event_observed=events)

print(kmf.survival_function_)
print(kmf.median_)
kmf.plot()
