import plotly.express as px
import pandas as pd
import pathlib
import numpy as np
import plotly.graph_objects as go


thisdir = pathlib.Path(__file__).parent.resolve()

def main():
    # Load data
    df = pd.read_csv(thisdir / "data" / "results" / "makespan.csv")
    df_bw = pd.read_csv(thisdir / "data" / "results" / "bandwidth.csv")
    df_bw_one = pd.read_csv(thisdir / "data" / "results" / "bandwidth_one.csv")
    df_num_execs = pd.read_csv(thisdir / "data" / "results" / "num_execs.csv")
    df_schedule_times = pd.read_csv(thisdir / "data" / "results" / "schedule_times.csv")

    n_sims = df['Simulation'].max()
    perimeter = 500
    n_trips = 3
    
    plotsdir = thisdir.joinpath('data', 'plots')
    plotsdir.mkdir(exist_ok=True, parents=True)
    plotsdir.joinpath('images').mkdir(exist_ok=True, parents=True)
    plotsdir.joinpath('html').mkdir(exist_ok=True, parents=True)
    facet_max = 4
    fig_width, fig_height = 1000, 600
    color_map = { # generated using https://davidmathlogic.com/colorblind/
        "HEFT": "#D81B60",
        "GCN": "#1E88E5",
        "Random": "#FFC107",
        "Static": "#004D40",
    }

    # Makespan Plot
    fig = px.scatter(
        df[df['Simulation'] <= facet_max], 
        x='Time', y='Makespan', 
        color='Scheduler', 
        facet_col='Simulation', 
        facet_col_wrap=int(np.sqrt(min(n_sims, facet_max))),
        template='plotly_white',
        trendline='expanding',
        color_discrete_map=color_map
        # trendline_options=dict(window=3)
    )
    fig.update_layout(width=fig_width, height=fig_height).write_image(str(plotsdir.joinpath('images', 'makespan.png')))
    fig.write_html(str(plotsdir.joinpath('html', 'makespans.html')))

    # Bandwidth Plot
    df_bw_sample = df_bw[(df_bw['Simulation'] <= facet_max) & (df_bw['Time'] <= perimeter)]
    fig_bw = px.line(
        df_bw_sample,
        x='Time', y='Avg Bandwidth', 
        # error_y='Std Bandwidth',
        facet_col='Simulation', 
        facet_col_wrap=int(np.sqrt(min(n_sims, facet_max))),
        template='plotly_white',
        color_discrete_map=color_map,
        labels={
            'Avg Bandwidth': 'Average Communication Rate',
            'Std Bandwidth': 'Standard Deviation of Communication Rate',
        }
    )
    # Add error band 
    gridwidth = int(np.sqrt(min(n_sims, facet_max)))
    for i_sim in range(1, min(n_sims, facet_max)+1):
        row = gridwidth - ((i_sim-1) // gridwidth)
        col = (i_sim-1) % gridwidth + 1
        _df = df_bw_sample[df_bw_sample['Simulation'] == i_sim]
        fig_bw.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=_df['Time'],
                y=_df['Avg Bandwidth'] + _df['Std Bandwidth'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        fig_bw.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=_df['Time'],
                y=_df['Avg Bandwidth'] - _df['Std Bandwidth'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ),
            row=row, col=col
        )

    fig_bw.update_traces(line_color='rgba(0,0,0,0.6)')
    fig_bw.write_image(str(plotsdir.joinpath('images', 'bandwidth.png')))
    fig_bw.write_html(str(plotsdir.joinpath('html', 'bandwidth.html')))

    # Single Node Bandwidth Plot
    df_bw_one_sample = df_bw_one[(df_bw_one['Simulation'] <= facet_max) & (df_bw_one['Time'] <= perimeter)]
    fig_bw_one = px.line(
        df_bw_one_sample,
        x='Time', y='Avg Bandwidth',
        # error_y='Std Bandwidth',
        facet_col='Simulation',
        facet_col_wrap=int(np.sqrt(min(n_sims, facet_max))),
        template='plotly_white',
        labels={
            'Avg Bandwidth': 'Average Communication Rate',
            'Std Bandwidth': 'Standard Deviation of Communication Rate',
        }
    )
    # Add error band
    gridwidth = int(np.sqrt(min(n_sims, facet_max)))
    for i_sim in range(1, min(n_sims, facet_max)+1):
        row = gridwidth - ((i_sim-1) // gridwidth)
        col = (i_sim-1) % gridwidth + 1
        _df = df_bw_one_sample[df_bw_one_sample['Simulation'] == i_sim]
        fig_bw_one.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=_df['Time'],
                y=_df['Avg Bandwidth'] + _df['Std Bandwidth'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=col
        )
        fig_bw_one.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=_df['Time'],
                y=_df['Avg Bandwidth'] - _df['Std Bandwidth'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ),
            row=row, col=col
        )

    fig_bw_one.update_traces(line_color='rgba(0,0,0,0.6)')
    fig_bw_one.write_image(str(plotsdir.joinpath('images', 'bandwidth_one.png')))
    fig_bw_one.write_html(str(plotsdir.joinpath('html', 'bandwidth_one.html')))

    # Schedule Times
    fig_schedule_times = px.violin(
        df_schedule_times,
        x='Scheduler', y='Time',
        template='plotly_white',
        labels={
            'value': 'Time (s)',
            'variable': 'Scheduler'
        }
    )
    fig_schedule_times.update_traces(line_color='rgba(0,0,0,0.6)')
    fig_schedule_times.write_image(str(plotsdir.joinpath('images', 'schedule_times.png')))
    fig_schedule_times.write_html(str(plotsdir.joinpath('html', 'schedule_times.html')))

    # Average Makespans
    df_mean = df.drop(columns=['Time']).groupby(['Scheduler', 'Simulation']).mean().reset_index()
    df_mean = df_mean.set_index('Simulation').sort_index()
    df_mean = df_mean.pivot_table(values='Makespan', index='Simulation', columns='Scheduler')
    df_mean = df_mean.div(df_mean['HEFT'], axis=0)
    
    fig = go.Figure()
    for scheduler in ['GCN', 'Static', 'Random']:
        fig.add_trace(go.Violin(
            y=df_mean[scheduler],
            name=scheduler,
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
            # fillcolor=hex_to_rgba_str(color_map[scheduler], 0.4),
            # line_color=color_map[scheduler]
            fillcolor='rgba(0,0,0,0.4)',
            line_color='rgba(0,0,0,0.6)'
        ))
    fig.update_layout(
        template='plotly_white',
        yaxis_title='Makespan Ratio',
        xaxis_title='Scheduler',
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_zeroline=False
    )
    fig.update_layout(width=fig_width, height=fig_height).write_image(str(plotsdir.joinpath('images', 'makespan_mean.png')))
    fig.write_html(str(plotsdir.joinpath('html', 'makespan_mean.html')))

    # Average Number of Executions
    df_num_execs_mean = df_num_execs.drop(columns=['Simulation']).mean().reset_index()
    df_num_execs_mean.columns = ['Scheduler', 'Average Number of Executions']
    fig_num_execs = px.bar(
        df_num_execs_mean,
        x='Scheduler', y='Average Number of Executions',
        template='plotly_white'
    )
    fig_num_execs.write_image(str(plotsdir.joinpath('images', 'num_execs.png')))
    fig_num_execs.write_html(str(plotsdir.joinpath('html', 'num_execs.html')))
    
    fig.update_layout(width=fig_width, height=fig_height).write_image(str(plotsdir.joinpath('images', 'makespan_mean.png')))
    fig.write_html(str(plotsdir.joinpath('html', 'makespan_mean.html')))

    # Number of Executions
    df_num_execs[['GCN', 'HEFT', 'Static', 'Random']] /= n_trips
    fig_num_execs = px.violin(
        df_num_execs,
        y=['HEFT', 'GCN', 'Static', 'Random'],
        template='plotly_white',
        box=True,
        labels={
            'value': 'Number of Executions',
            'variable': 'Scheduler'
        }
    )
    fig_num_execs.update_traces(
        line_color='rgba(0,0,0,0.6)', 
        fillcolor='rgba(0,0,0,0.4)',
        marker_color='rgba(0,0,0,0.6)'
    )
    fig_num_execs.write_image(str(plotsdir.joinpath('images', 'num_execs.png')))
    fig_num_execs.write_html(str(plotsdir.joinpath('html', 'num_execs.html')))

if __name__ == '__main__':
    main()