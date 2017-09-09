from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('ticks')
sns.set_palette('Paired', 10)
sns.set_context('talk')
sns.set(font_scale=2)

# Helper function to plot results.
def plot_results(times, title):
    plt.figure(figsize=(15, 8))
    xstart = 0.0
    ticks = []
    labels = []
    for (scheme, time) in times.items():
        plt.bar(xstart, time, label=scheme)
        labels.append(scheme)
        ticks.append(xstart)
        xstart += 0.8
    plt.title(title)
    plt.ylabel("Time (in seconds)")
    plt.xlabel("Scheme")
    plt.xticks(ticks, labels)
    plt.show()
