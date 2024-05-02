import plot
import trough_identify


fig, ax, all_matched_doublets, object_num, data = plot.plot_object()

plot.save_results(fig, ax, all_matched_doublets, object_num, data)
