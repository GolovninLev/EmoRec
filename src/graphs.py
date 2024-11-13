from io import BytesIO

import plotly.graph_objects as go
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use


colors = ["#e74c3c", "#a93226", "#27ae60", "#444444", "#f39c12", "#999999", "#3498db", "#9b59b6"]


def generate_emotion_map_html(data, labels, input_video_fps, num_ticks=15):

    data = np.array(data)
    time_ticks_num = np.arange(data.shape[0])


    fig = go.Figure(data=[go.Scatter(
        x=time_ticks_num,
        y=data[:, i],
        mode='lines',
        line=dict(color=colors[i]),
        stackgroup='one',
        name=labels[i],
        opacity=0.84,
        fillcolor=colors[i],  # добавляем цвет под графиком
        fill='tonexty'  # указываем, что цвет должен заполнять область под графиком tozeroy tonexty toself
    ) for i in range(data.shape[1])])
    fig.update_layout(title_text="Карта эмоций на видео")


    # Оси с динамическими максимальными значениями
    max_value = np.max(np.sum(data, axis=1))
    fig.update_layout(
        yaxis=dict(range=[0, max_value]),
        xaxis=dict(range=[0, data.shape[0] - 0.001])
    )


    # Метки по оси Х
    time_per_frame = 1 / input_video_fps
    tick_indices = np.linspace(0, time_ticks_num.shape[0], num_ticks, endpoint=True)
    tick_labels = []
    axis_label = ""
    
    for i in tick_indices:
        time_in_seconds = i * time_per_frame
        h = int(time_in_seconds // 3600)
        m = int((time_in_seconds % 3600) // 60)
        s = int(time_in_seconds % 60)
        ds = int((time_in_seconds % 1) * 10)
        if h == 0:
            tick_labels.append(f"{m:02d}:{s:02d}")
            axis_label = "Время (мин:сек)"
        else:
            tick_labels.append(f"{h:02d}:{m:02d}:{s:02d}")
            axis_label = "Время (час:мин:сек)"
    
    fig.update_layout(
        xaxis=dict(ticktext=tick_labels, tickvals=tick_indices, tickangle=45),
        xaxis_title=axis_label
    )
    
    fig.update_layout(
        xaxis=dict(
            tickformatstops=[
                dict(dtickrange=[None, 60], value="%S"),
                dict(dtickrange=[60, 3600], value="%M:%S"),
                dict(dtickrange=[3600, None], value="%H:%M:%S")
            ],
        ),
    )


    # Метки по оси Y
    yticks = np.array([i * max_value / 100 for i in range(0, 110, 10)])
    ytick_labels = [f"{i}%" for i in range(0, 110, 10)]
    
    fig.update_layout(
        yaxis=dict(ticktext=ytick_labels, tickvals=yticks)
    )


    # Легенда графика
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))


    # Возврат BytesIO файла
    buf = BytesIO()
    html_str = fig.to_html(include_plotlyjs='cdn')
    buf = html_str.encode('utf-8')
    return buf



def generate_emotion_map_png(data, labels, input_video_fps, num_ticks=15):
    
    mpl_use('Agg')
    
    data = np.array(data)
    time_ticks_num = np.arange(data.shape[0])


    plt.figure(figsize=(30, 6))
    plt.stackplot(time_ticks_num, data.T, colors=colors)
    plt.suptitle("Карта эмоций на видео")


    # Оси с динамическими максимальными значениями
    max_value = np.max(np.sum(data, axis=1))
    plt.ylim(0, max_value)
    plt.xlim(0, data.shape[0] - 0.001)


    # Метки по оси Х
    time_per_frame = 1 / input_video_fps
    tick_indices = np.linspace(0, time_ticks_num.shape[0], num_ticks, endpoint=True)
    tick_labels = []
    axis_label = ""
    
    for i in tick_indices:
        time_in_seconds = i * time_per_frame
        h = int(time_in_seconds // 3600)
        m = int((time_in_seconds % 3600) // 60)
        s = int(time_in_seconds % 60)
        ds = int((time_in_seconds % 1) * 10)
        if h == 0:
            tick_labels.append(f"{m:02d}:{s:02d}")
            axis_label = "Время (мин:сек)"
        else:
            tick_labels.append(f"{h:02d}:{m:02d}:{s:02d}")
            axis_label = "Время (час:мин:сек)"
    
    plt.xticks(tick_indices, tick_labels, rotation=45)
    plt.xlabel(axis_label)


    # Метки по оси Y
    yticks = np.array([i * max_value / 100 for i in range(0, 110, 10)])
    ytick_labels = [f"{i}%" for i in range(0, 110, 10)]
    
    plt.yticks(yticks, ytick_labels)
    plt.ylabel("Проценты")


    # Легенда графика
    plt.legend([l.replace('\U0001F922', '\U0001F612') for l in labels], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 1.03])  # добавление отступа сверху


    # Возврат BytesIO файла
    buf = BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    return buf
