import os
import tkinter as tk
import math
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import ast
from scipy.spatial.distance import cdist
# linear interpolation
from scipy.spatial.distance import cdist


class SquareGrid(tk.Canvas):
    def __init__(self, master, size, scale_factor):
        super().__init__(master, width=size * scale_factor, height=size * scale_factor)
        self.size = size
        self.scale_factor = scale_factor
        self.node_coords = {}
        self.node_colors = {}
        self.node_ranges = {}
        self.node_selector = None
        self.marker_radius = 5
        self.header = ['node', 'x', 'y', 'time']
        self.trace = []

        # Draw square grid
        for i in range(10):
            for j in range(10):
                x1 = i * (size * scale_factor) / 10
                y1 = j * (size * scale_factor) / 10
                x2 = (i + 1) * (size * scale_factor) / 10
                y2 = (j + 1) * (size * scale_factor) / 10
                self.create_rectangle(x1, y1, x2, y2)

        # Add scale label
        scale_label = tk.Label(master, text=f"Scale: {(size * scale_factor)} square meters per square.\n All nodes "
                                            f"must be placed at a given time period for mapping network state's "
                                            f"purposes.")
        scale_label.pack()

        # Bind left mouse button click event
        self.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        node_id = self.node_selector.get()
        x, y = event.x, event.y
        # x, y = event.x * self.scale_factor, event.y * self.scale_factor
        time = self.time_entry.get()
        self.create_oval(x - self.marker_radius, y - self.marker_radius,
                         x + self.marker_radius, y + self.marker_radius,
                         fill=self.node_colors[node_id])
        self.create_oval(x - self.node_ranges[node_id], y - self.node_ranges[node_id],
                         x + self.node_ranges[node_id], y + self.node_ranges[node_id],
                         outline=self.node_colors[node_id])
        self.node_coords[node_id] = (x * self.scale_factor, y * self.scale_factor, time)
        print(f"Node ID: {node_id}, X: {x * self.scale_factor}, Y: {y * self.scale_factor}, Time: {time}")
        self.trace.append([str(node_id).replace('Node ', ''), x * self.scale_factor, y * self.scale_factor, time])
        with open(save_to_folder + save_to_file, mode='w', newline='') as trace_file:
            trace_writer = csv.writer(trace_file)
            trace_writer.writerow(self.header)
            trace_writer.writerows(self.trace)

    def set_node_selector(self, options):
        self.node_selector = tk.StringVar(value=options[0])
        selector = tk.OptionMenu(self.master, self.node_selector, *options)
        selector.pack()

    def set_node_colors(self, colors):
        self.node_colors = colors

    def set_node_ranges(self, ranges):
        self.node_ranges = ranges

    def set_marker_radius(self, radius):
        self.marker_radius = radius

    def set_time_entry(self, entry):
        self.time_entry = entry


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)


# creating n points between x and y using linear interpolation
def get_equidistant_points(trace, n):
    node_id = trace.loc[0, 'node']
    new_trace = pd.DataFrame(columns=trace.columns.values)
    for line in range(1, len(trace)):
        x = trace.loc[line - 1, 'x']
        y = trace.loc[line - 1, 'y']
        t = trace.loc[line - 1, 'time']

        x_1 = trace.loc[line, 'x']
        y_1 = trace.loc[line, 'y']
        t_1 = trace.loc[line, 'time']

        time_interval = float(t_1) - float(t)
        current_time = float(t)
        for i in range(n):
            p1 = lerp(float(x), float(x_1), 1. / n * i)
            p2 = lerp(float(y), float(y_1), 1. / n * i)

            # new_trace = new_trace.append({'node': node_id, 'x': p1, 'y': p2, 'time': current_time}, ignore_index=True)
            new_trace = pd.concat(
                [new_trace, pd.DataFrame({'node': [node_id], 'x': p1, 'y': p2, 'time': current_time})],
                ignore_index=True)

            # trace_temp = pd.DataFrame.from_dict({'node': node_id, 'x': p1, 'y': p2, 'time': current_time})
            # new_trace = pd.concat([new_trace, trace_temp], axis= 0)

            add_time = time_interval / n
            current_time = current_time + add_time

        # filling the last state
        if line == len(trace) - 1:
            # new_trace = new_trace.append({'node': node_id, 'x': x_1, 'y': y_1, 'time': t_1}, ignore_index=True)
            new_trace = pd.concat([new_trace, pd.DataFrame({'node': [node_id], 'x': [x_1], 'y': [y_1], 'time': [t_1]})],
                                  ignore_index=True)
            # trace_temp = pd.DataFrame.from_dict({'node': node_id, 'x': x_1, 'y': y_1, 'time': t_1})
            # new_trace = pd.concat([new_trace, trace_temp], axis= 0)

    return new_trace


# create folder
def creatingFolders(dataFolder):
    if (os.path.isdir(dataFolder) == False):
        os.makedirs(dataFolder)


# reading trace files
def get_trace(trace_file, csv):
    if csv == "csv":
        trace = pd.read_csv(trace_file, sep=',')
        trace = trace[['node', 'x', 'y', 'time']]
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


# getting the correspondent state based on a location
def datarate_match(trace, rings, trace2):
    nodeList = trace[['x', 'y']]  # trace.iloc[:, 1:3]
    # node = [[str(center), str(center)]]
    # node = [[(center[0]), (center[1])]]
    nodeList2 = trace2[['x', 'y']]

    # for line in range(0, len(trace)):
    #    col = trace[line].split(",")
    #    nodeList.append([col[0],col[1]])
    # node.append([str(center), str(center)])
    # dist = scipy.spatial.distance.cdist(node,nodeList)
    node_distance = cdist(nodeList, nodeList2, 'euclidean')
    node_datarate = []
    i = 0
    for dist in node_distance:
        for key in rings:
            lim_inf = float(rings[key].split(",")[0])
            lim_sup = float(rings[key].split(",")[1])
            if lim_inf <= dist <= lim_sup:
                node_datarate.append(key)
                break

        print(str(i) + ' state ' + str(node_datarate[i]) + ' distance ' + str(dist))
        i = i + 1

    trace['state'] = node_datarate
    # trace['dist_nodes'] = node_distance
    # print(trace)
    return trace


def compute_distance(node1, node2):
    # x1, y1, time1 = node1
    # x2, y2, time2 = node2
    distance = math.sqrt((node2[1] - node1[1]) ** 2 + (node2[2] - node1[2]) ** 2)
    return distance


def add_datarate(trace, communicatuon_dic):
    new_trace = pd.DataFrame(columns=['node', 'x', 'y', 'time', 'states'])
    trace_node_temp = trace.groupby('time')
    last_key = (9999, 9999)

    for n in trace_node_temp.groups:
        trace = trace_node_temp.get_group(n).reset_index()
        trace.drop(['index'], axis='columns', inplace=True)
        # Dictionary to store the distances between nodes
        distances = {}
        node_datarate = {}
        for i in range(len(trace)):
            node1 = trace.iloc[i]
            node_states = []
            for j in range(i + 1, len(trace)):
                node2 = trace.iloc[j]
                distance = compute_distance(node1, node2)
                # strategy to have unique first key. Ex. (1,2), (1,3) -> (1,2), (3,1)
                key = (int(node1[0]), int(node2[0]))
                if (last_key[0] == int(node1[0])) & (key[0] == int(node1[0])) & (last_key[1] != int(node1[1])):
                    key = (int(node2[0]), int(node1[0]))
                last_key = key

                # key = str(int(node1[0]))+'_'+str(int(node2[0]))
                distances[key] = np.round(distance, decimals=2)

                for key_rings in communicatuon_dic:
                    lim_inf = float(communicatuon_dic[key_rings].split(",")[0])
                    lim_sup = float(communicatuon_dic[key_rings].split(",")[1])
                    if lim_inf <= distances[key] <= lim_sup:
                        node_datarate[key] = key_rings
                        break
                # node_states.append(node_datarate)
                # n1, n2 = key
                print(
                    f"The distance between node {node1[0]} and node {node2[0]} at time {node1[3]} is {distances[key]} and state {node_datarate[key]}")

            new_trace = pd.concat([new_trace, pd.DataFrame(
                {'node': int(node1[0]), 'x': node1[1], 'y': node1[2], 'time': node1[3], 'states': [node_datarate]})],
                                  ignore_index=True)

    return new_trace


def create_animation(data, img_folder,file_description):

    data_transformed = data.copy()
    # unit in 'Kilometers'
    data_transformed['x'] = data_transformed['x'].apply(lambda x: x / 1000)
    data_transformed['y'] = data_transformed['y'].apply(lambda x: x / 1000)

    # defining the graphic bounds
    x_lim = int(data_transformed['x'].max()) + 2
    y_lim = int(data_transformed['y'].max()) + 2

    groups = data_transformed.groupby('time')

    #fig = plt.figure()  # figsize=(6, 6))


    # fig, ax = plt.subplots()
    # fig.set_size_inches(6, 6, forward=True)

    state_color = ['#CC0000', '#FFE66C', '#EBD367', '#D0B100', '#2B8C48', '#005E25']

    if len(groups.groups[0]) >= 3:
        # 3 different nodes
        #c = ["#0000FF", "#00FF00", "#FF0066"]
        m = ["o", "^", "s"]
        l = ["VHF", "UHF", "SatCom"]
        legend_labels = ['VHF', 'UHF', 'SatCom']  # Replace with your actual legend labels
    if len(groups.groups[0]) == 2:
        # 2 nodes
        m = ["^", "^", "s"]
        l = ["UHF", "UHF"]
        legend_labels = ['UHF']  # Replace with your actual legend labels


    legend_states = ['0', '1', '2', '3', '4', '5']  # Replace with your actual legend labels

    fig, ax = plt.subplots()
    fig.set_size_inches(6.6, 5.6)
    for name, group in groups:
        for i in range(len(group)):

            # Transforming the dictionary {(1, 2): 4, (1, 3): 2, (2, 3): 3} into {(1): 4, (3): 2, (2): 3},
            # due to different matching. (1, 2) -> node 1, (1, 3) -> node 3, (2, 3) -> node 2
            #state_dic = ast.literal_eval(group.iloc[i].states)
            state_dic = group.iloc[i].states
            node_datarate = {}
            for key, value in state_dic.items():
                if key[0] in node_datarate:
                    node_datarate[key[1]] = value
                else:
                    node_datarate[key[0]] = value

            #node_datarate = {key[0]: value for key, value in state_dic.items()}
            if len(group) != 2:
                ax.plot(group.iloc[i].x, group.iloc[i].y, linestyle='', marker=m[int(group.iloc[i].node) - 1], ms=10,
                        color=state_color[node_datarate[int(group.iloc[i].node)]])  # c[int(group.iloc[i].node) - 1]
            else:
                ax.plot(group.iloc[i].x, group.iloc[i].y, linestyle='', marker=m[int(group.iloc[i].node) - 1], ms=10,
                        color=state_color[node_datarate[int(group.iloc[0].node)]])  # c[int(group.iloc[i].node) - 1]

            if not track:
                annotation = ax.annotate(l[int(group.iloc[i].node) - 1], xy=(group.iloc[i].x, group.iloc[i].y),
                                         xytext=(group.iloc[i].x - 0.5, group.iloc[i].y + 0.5),
                                         arrowprops={'arrowstyle': "->"}, fontsize=12)
                annotation.set_animated(True)

        ax.set_xlim([0, x_lim])
        ax.set_xlabel('Kilometers', fontsize=20)
        ax.set_ylim([0, y_lim])
        ax.set_ylabel('Kilometers', fontsize=20)
        # ax.set_title(f'Time: {name} seconds', fontsize=14)

        plt.yticks(fontsize=18, rotation=0)
        plt.xticks(fontsize=18, rotation=0)
        plt.grid(color='gray', linestyle='dashed')
        #fig.tight_layout()
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_title(f'{name} sec', fontsize=10,loc='right')

        # # Add legend
        # scatter_elements = [plt.Line2D([0], [0], marker=m[i], color='w', markerfacecolor='gray', markersize=8) for i in
        #                     range(len(legend_labels))]
        # node_legend = ax.legend(scatter_elements, legend_labels, loc='lower right', title="Radio Type")
        # state_elements = [plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=state_color[i], markersize=8) for
        #                   i in
        #                   range(len(legend_states))]
        # network_legend = ax.legend(state_elements, legend_states, loc='upper right', title="Network\nStates")
        #
        # ax.add_artist(node_legend)
        # ax.add_artist(network_legend)

        # Add legend
        scatter_elements = [plt.Line2D([0], [0], marker=m[i], color='w', markerfacecolor='gray', markersize=12) for i in
                            range(len(legend_labels))]
        node_legend = ax.legend(scatter_elements, legend_labels, fontsize=14,
                                loc='lower right')  # , title="Radio Type")
        state_elements = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=state_color[i], markersize=16) for
                          i in range(len(legend_states))]
        network_legend = ax.legend(state_elements, legend_states, loc='upper center', bbox_to_anchor=(0.5, 1.16),
                                   columnspacing=.5,
                                   fontsize=16, ncol=6, handletextpad=-.5, fancybox=False,
                                   shadow=False)  # , title="Network States")

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
                    duration=300)  # Use `duration`(in ms) instead, e.g. `fps=50` == `duration=20` (1000 * 1/50).

    for name, group in groups:
        if os.path.exists(img_folder + f'img_{name}.png'):
            os.remove(img_folder + f'img_{name}.png')
        else:
            print("The file does not exist")

def create_static_plot(data, img_folder, file_description):


    data_transformed = data.copy()
    # unit in 'Kilometers'
    data_transformed['x'] = data_transformed['x'].apply(lambda x: x / 1000)
    data_transformed['y'] = data_transformed['y'].apply(lambda x: x / 1000)

    # defining the graphic bounds
    x_lim = int(data_transformed['x'].max()) + 1
    y_lim = int(data_transformed['y'].max()) + 1

    groups = data_transformed.groupby('time')

    #fig = plt.figure()  # figsize=(6, 6))

    # fig, ax = plt.subplots()
    # fig.set_size_inches(6, 6, forward=True)

    state_color = ['#CC0000', '#FFE66C', '#EBD367', '#D0B100', '#2B8C48', '#005E25']

    # 3 different nodes
    #c = ["#0000FF", "#00FF00", "#FF0066"]
    m = ["o", "^", "s"]
    l = ["VHF", "UHF", "SatCom"]
    legend_labels = ['VHF', 'UHF', 'SatCom']  # Replace with your actual legend labels

    # 2 nodes
    # m = ["^", "^", "s"]
    # l = ["UHF", "UHF"]
    # legend_labels = ['UHF']  # Replace with your actual legend labels


    legend_states = ['0', '1', '2', '3', '4', '5']  # Replace with your actual legend labels

    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 4.5)

    for name, group in groups:
        for i in range(len(group)):

            # Transforming the dictionary {(1, 2): 4, (1, 3): 2, (2, 3): 3} into {(1): 4, (3): 2, (2): 3},
            # due to different matching. (1, 2) -> node 1, (1, 3) -> node 3, (2, 3) -> node 2
            #state_dic = ast.literal_eval(group.iloc[i].states)
            state_dic = group.iloc[i].states
            #node_datarate = {key[0]: value for key, value in state_dic.items()}
            node_datarate = {}
            for key, value in state_dic.items():
                if key[0] in node_datarate:
                    node_datarate[key[1]] = value
                else:
                    node_datarate[key[0]] = value

            ax.plot(group.iloc[i].x, group.iloc[i].y, linestyle='', marker=m[int(group.iloc[i].node) - 1], ms=4,
                    color=state_color[node_datarate[int(group.iloc[i].node)]])  # c[int(group.iloc[i].node) - 1]

    #plt.subplots_adjust(bottom=0.30)
    #plt.subplots_adjust(top=1)
    #plt.subplots_adjust(right=1)
    ax.set_xlim([0, x_lim])
    ax.set_xlabel('Kilometers', fontsize=20)
    ax.set_ylim([0, y_lim])
    ax.set_ylabel('Kilometers', fontsize=20)
    #ax.set_title(f'Time: {name} seconds', fontsize=14)

    plt.yticks(fontsize=18, rotation=0)
    plt.xticks(fontsize=18, rotation=0)
    plt.grid(color='gray', linestyle='dashed')
    fig.tight_layout()
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * -0.05, box.width, box.height * 0.1])

    # Add legend
    scatter_elements = [plt.Line2D([0], [0], marker=m[i], color='w', markerfacecolor='gray', markersize=12) for i in
                        range(len(legend_labels))]
    node_legend = ax.legend(scatter_elements, legend_labels,fontsize=14, loc='lower right')#, title="Radio Type")
    state_elements = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=state_color[i], markersize=16) for
                      i in range(len(legend_states))]
    network_legend = ax.legend(state_elements, legend_states, loc='upper center', bbox_to_anchor=(0.5, 1.16), columnspacing=.5,
                  fontsize=16, ncol=6, handletextpad=-.5, fancybox=False, shadow=False)#, title="Network States")

    ax.add_artist(node_legend)
    ax.add_artist(network_legend)



    plt.savefig(img_folder + file_description+'.pdf', format='pdf', transparent=False, facecolor='white',bbox_inches='tight', dpi=300)

    plt.close(fig)






if __name__ == "__main__":
    save_to_folder = os.path.dirname(os.path.abspath(__file__)) + '/data/drawing_trace/'
    img_folder = os.path.dirname(os.path.abspath(__file__)) + '/data/img/'

    #save_to_file = 'trace_pendulum.csv'  # 'trace_convoy_3V.csv'
    #save_to_file = 'trace_convoy_no_disconnections.csv'
    #save_to_file = 'trace_convoy_disconnections.csv'
    #save_to_file = 'trace_M5.csv'
    #save_to_file = 'trace.csv'  # 'trace_convoy_3V.csv'

    save_to_file = 'trace_convoy_disconnections2.csv'

    creatingFolders(save_to_folder)

    track = False  # set the nodes track to show in the gif file


    '''
    Creating the graphic interface to draw a mobility pattern
    '''

    root = tk.Tk()

    # Set up the SquareGrid widget
    square_size = 100  # meters
    window_size = 1000  # pixels
    # Calculate the scale factor
    scale_factor = window_size / square_size
    square_grid = SquareGrid(root, size=square_size, scale_factor=scale_factor)
    square_grid.pack(side="left")

    # Set up the node selector
    nodes = ["Node 1", "Node 2", "Node 3"]
    square_grid.set_node_selector(nodes)

    # Set up the node colors
    colors = {"Node 1": "red", "Node 2": "blue", "Node 3": "green"}
    square_grid.set_node_colors(colors)

    # Set up the node communication ranges
    # 100 = 1 km square
    ranges = {"Node 1": 200, "Node 2": 200, "Node 3": 200}
    square_grid.set_node_ranges(ranges)

    # Set up the marker radius
    square_grid.set_marker_radius(5)

    # Set up the time entry widget
    time_entry = tk.Entry(root)
    time_entry.pack()
    square_grid.set_time_entry(time_entry)

    root.mainloop()

    '''
    Creating the link state using the trace drawing tool
    '''

    # defines the radius distance (10m, 100m,1000m...)
    # #magnitude = 200  # 10-WIFI  200-UHF 2000-VHF
    datarate_rings_uhf = dict(
        [(5, "0.0," + str(2 * 200)),
         (4, str(2 * 200 + 0.00001) + "," + str(4 * 200)),
         (3, str(4 * 200 + 0.00001) + "," + str(6 * 200)),
         (2, str(6 * 200 + 0.00001) + "," + str(8 * 200)),
         (1, str(8 * 200 + 0.00001) + "," + str(10 * 200)),
         (0, str(10 * 200 + 0.00001) + "," + str(50 * 200))])

    datarate_rings_vhf = dict(
        [(5, "0.0," + str(2 * 2000)),
         (4, str(2 * 2000 + 0.00001) + "," + str(4 * 2000)),
         (3, str(4 * 2000 + 0.00001) + "," + str(6 * 2000)),
         (2, str(6 * 2000 + 0.00001) + "," + str(8 * 2000)),
         (1, str(8 * 2000 + 0.00001) + "," + str(10 * 2000)),
         (0, str(10 * 2000 + 0.00001) + "," + str(50 * 2000))])

    datarate_rings_satcom = dict(
        [(5, "0.0," + str(2 * 20000)),
         (4, str(2 * 20000 + 0.00001) + "," + str(4 * 20000)),
         (3, str(4 * 20000 + 0.00001) + "," + str(6 * 20000)),
         (2, str(6 * 20000 + 0.00001) + "," + str(8 * 20000)),
         (1, str(8 * 20000 + 0.00001) + "," + str(10 * 20000)),
         (0, str(10 * 20000 + 0.00001) + "," + str(50 * 20000))])
    #
    trace_data = get_trace(save_to_folder + save_to_file, 'csv')

    trace_nodes = pd.DataFrame()
    trace_node_temp = trace_data.groupby('node')
    for n in trace_node_temp.groups:
        trace = trace_node_temp.get_group(n).reset_index()
        trace.drop(['index'], axis='columns', inplace=True)

        trace = get_equidistant_points(trace, 10)
        # trace = add_dist_speed(trace)

        trace_nodes = pd.concat([trace_nodes, trace])

    trace_nodes = add_datarate(trace_nodes, datarate_rings_uhf)

    trace_nodes.to_csv(save_to_folder + save_to_file.replace(".csv", "_states") + '.csv', index=False)

    '''
    Creating a gif with the trace file
    '''

    scenario = save_to_file.replace('.csv','_states.csv')
    # scenario = 'trace_pendulum_states.csv'
    file_description = scenario.replace('.csv', '')
    #df = get_trace(save_to_folder + scenario, 'csv')

    if track:
        scenario = scenario.replace('.csv', '_track.csv')

    creatingFolders(img_folder)


    create_static_plot(trace_nodes, img_folder, file_description)
    create_animation(trace_nodes, img_folder, file_description)