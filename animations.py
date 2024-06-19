import os
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import glob
from PIL import Image
import numpy as np
import ast


# create folder
def creatingFolders(dataFolder):
    if (os.path.isdir(dataFolder) == False):
        os.makedirs(dataFolder)


# reading trace files
def get_trace(trace_file, csv):
    if csv == "csv":
        trace = pd.read_csv(trace_file, sep=',')
        trace = trace[['node', 'x', 'y', 'time', 'states']]
    else:
        file_ = open(trace_file, 'r')
        raw_data = file_.readlines()
        file_.close()

        x = []
        y = []
        t = []
        node = []
        # trace = []
        for line in range(0, len(raw_data)):
            col = raw_data[line].split(" ")
            for elem in range(0, len(col), 3):
                t.append(col[elem])
                x.append(float(col[elem + 1]))
                y.append(float(col[elem + 2]))
                node.append(int(line))
                # trace.append(str(col[elem+1]) + "," + str(col[elem+2]) + "," + str(col[elem]))

        # Create a DataFrame object
        trace = pd.DataFrame(columns=['node', 'x', 'y', 'time'])
        trace['node'] = node
        trace['x'] = x
        trace['y'] = y
        trace['time'] = t

    # trace = pd.DataFrame(dict(x=x, y=y, time=t))

    return (trace)


def create_animation(data, img_folder):
    # defining the graphic bounds
    x_lim = int(data['x'].max()) + 2000
    y_lim = int(data['y'].max()) + 2000

    groups = data.groupby('time')

    fig = plt.figure()  # figsize=(6, 6))

    # fig, ax = plt.subplots()
    # fig.set_size_inches(6, 6, forward=True)

    state_color = ['#CC0000', '#FFE66C', '#EBD367', '#D0B100', '#2B8C48', '#005E25']
    c = ["#0000FF", "#00FF00", "#FF0066"]
    m = ["o", "^", "s"]
    l = ["VHF", "UHF", "SatCom"]
    legend_labels = ['VHF', 'UHF', 'SatCom']  # Replace with your actual legend labels
    legend_states = ['0', '1', '2', '3', '4', '5']  # Replace with your actual legend labels

    fig, ax = plt.subplots()

    for name, group in groups:
        for i in range(len(group)):

            # Transforming the dictionary {(1, 2): 4, (1, 3): 2, (2, 3): 3} into {(1): 4, (3): 2, (2): 3},
            # due to different matching. (1, 2) -> node 1, (1, 3) -> node 3, (2, 3) -> node 2
            state_dic = ast.literal_eval(group.iloc[i].states)
            # node_datarate = {}
            # for key, value in state_dic.items():
            #     if key[0] in node_datarate:
            #         node_datarate[key[1]] = value
            #     else:
            #         node_datarate[key[0]] = value

            node_datarate = {key[0]: value for key, value in state_dic.items()}

            ax.plot(group.iloc[i].x, group.iloc[i].y, linestyle='', marker=m[int(group.iloc[i].node) - 1], ms=6,
                    color=state_color[node_datarate[int(group.iloc[i].node)]])  # c[int(group.iloc[i].node) - 1]

            if not track:
                annotation = ax.annotate(l[int(group.iloc[i].node) - 1], xy=(group.iloc[i].x, group.iloc[i].y),
                                         xytext=(group.iloc[i].x - 1000, group.iloc[i].y + 1000),
                                         arrowprops={'arrowstyle': "->"})
                annotation.set_animated(True)

        ax.set_xlim([0, x_lim])
        ax.set_xlabel('X coordinate (meters)', fontsize=14)
        ax.set_ylim([0, y_lim])
        ax.set_ylabel('Y coordinate (meters)', fontsize=14)
        ax.set_title(f'Time: {name} seconds', fontsize=14)

        # Add legend
        scatter_elements = [plt.Line2D([0], [0], marker=m[i], color='w', markerfacecolor='gray', markersize=8) for i in
                            range(len(legend_labels))]
        node_legend = ax.legend(scatter_elements, legend_labels, loc='lower right', title="Radio Type")
        state_elements = [plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=state_color[i], markersize=8) for
                          i in
                          range(len(legend_states))]
        network_legend = ax.legend(state_elements, legend_states, loc='upper right', title="Network\nStates")

        ax.add_artist(node_legend)
        ax.add_artist(network_legend)

        plt.savefig(img_folder + f'img_{name}.png', transparent=False, facecolor='white')

        if not track:
            ax.clear()  # at the end of each iteration, we remove the plotted data but keep the axis settings intact for the next plot

    plt.close(fig)

    frames = []
    for name, group in groups:
        image = imageio.v2.imread(img_folder + f'img_{name}.png')
        frames.append(image)

    imageio.mimsave(img_folder + file_description + '.gif',  # output gif
                    frames,  # array of input frames
                    loop=0, quality=100, optimize=False, save_all=True, transparent=False,
                    facecolor='white',
                    duration=200)  # Use `duration`(in ms) instead, e.g. `fps=50` == `duration=20` (1000 * 1/50).

    for name, group in groups:
        if os.path.exists(img_folder + f'img_{name}.png'):
            os.remove(img_folder + f'img_{name}.png')
        else:
            print("The file does not exist")


if __name__ == "__main__":
    # app.run_server(debug=True)

    # read_from = os.path.dirname(os.path.abspath(__file__)) + '/data/bonn_motion/'
    read_from = os.path.dirname(os.path.abspath(__file__)) + '/data/drawing_trace/'
    img_folder = os.path.dirname(os.path.abspath(__file__)) + '/data/img/'

    track = False  # set the nodes track to show in the gif

    scenario = 'trace_states.csv'
    #scenario = 'trace_pendulum_states.csv'
    file_description = scenario.replace('.csv', '') # "Trace_Convoy_3_Nodes"  # "" # node to base station ("") or node to node ("_NtoN_")

    df = get_trace(read_from + scenario, 'csv')

    if track:
        scenario = scenario.replace('.csv', '_track.csv')

    creatingFolders(img_folder)

    create_animation(df, img_folder)
