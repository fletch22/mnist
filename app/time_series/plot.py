from matplotlib import pyplot


def show_time_series_chart(df, columns_indices_to_plot):
    values = df.values
    # specify columns to plot
    groups = columns_indices_to_plot
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

def show_performance(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.show()