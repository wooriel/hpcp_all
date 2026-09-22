from sklearn.manifold import TSNE

all_global_features
all_projected_features
all_work_ids

z = torch.cat(all_projected_features).cpu().numpy()
labels = torch.cat(all_work_ids).cpu().numpy()

z_2d = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
).fit_transform(z)

plt.scatter(
    z_2d[:, 0],
    z_2d[:, 1],
    c=labels,
    s=8,
)