import matplotlib.pyplot as plt
import numpy as np

class HoverHandler:
    CHAR_WIDTH = 40
    NEW_LINE = '\n' + '_' * CHAR_WIDTH + '\n'
    
    def __init__(self, fig, ax, sc, annotations, annot):
        self.fig = fig
        self.ax = ax
        self.sc = sc
        self.annotations = annotations
        self.annot = annot
        

    def update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = ''
        for n in ind["ind"]:
            item = f'[{n}] {self.annotations[n]}\n'
            
            ## Adding new lines every CHAR_WIDTH characters
            i = HoverHandler.CHAR_WIDTH
            while i < len(item):
                item = item[:i] + '\n' + item[i:]
                i+= HoverHandler.CHAR_WIDTH

            ## Concat multiple addnotations
            if len(text) > 0:
                text += HoverHandler.NEW_LINE
            text += item
            
        self.annot.set_text(text)
    

    def __call__(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

                    

def cluster_plot(embeddings,
                  cluster_assignment = None,
                  annotations = None,
                  outliers_mask = None,
                  centroids = None,
                  cmap='viridis',
                  sources=None):
    FS = (10, 8)
    fig, ax = plt.subplots(figsize=FS)
    N = embeddings.shape[0]

    if cluster_assignment is None:
        cluster_assignment = np.full((N,), 0)
    
    num_clusters = np.unique(cluster_assignment).shape[0]
    
    if outliers_mask is None:
        outliers_mask = np.full((N,), False)
        
    embs_out = embeddings[outliers_mask == True]
    cluster_assignment_out = cluster_assignment[outliers_mask == True]
    
    embs_tru = embeddings[outliers_mask == False]
    cluster_assignment_tru = cluster_assignment[outliers_mask == False]
    
    if sources is not None:
        color_map = {'TikTok': '#000000', 'YouTube': '#808080'}
        colors = np.array([color_map.get(src, '#808080') for src in sources])
    else:
        colors = np.array(['#808080'] * N)

    sc_all = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0, s=8)
    sc_tru = ax.scatter(embs_tru[:, 0], embs_tru[:, 1], c=colors[outliers_mask == False], alpha=1, s=8)
    sc_out = ax.scatter(embs_out[:, 0], embs_out[:, 1], c=colors[outliers_mask == True], alpha=0.2, marker='x', s=20)

    if centroids is not None:
        sc_ctr = ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='r')
        for idx, (x_ctr, y_ctr) in enumerate(centroids):
            ax.text(
                x_ctr,
                y_ctr,
                str(idx),
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                color='black'
            )



    ## Plotting legend
    if sources is not None:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='TikTok', markerfacecolor='#000000', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='YouTube', markerfacecolor='#808080', markersize=6),
        ]
        ax.legend(handles=legend_elements, loc="lower left", title="Source", fontsize=8)

    if annotations is not None:
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="-"))
        annot.set_visible(False)
        hover = HoverHandler(fig, ax, sc_all, annotations, annot)
        fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.axis('off')
    plt.show()
